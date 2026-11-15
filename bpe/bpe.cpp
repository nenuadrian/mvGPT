// Byte-level BPE: training merges on a text, and encoding text with them. Exposed to Python with pybind11.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Pair = std::pair<int, int>;

// Letters, digits, and the bytes of multi-byte UTF-8 characters, which stay inside words.
bool IsWord(unsigned char c) { return std::isalnum(c) || c >= 0x80; }
bool IsSpace(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// Splits text into chunks that merges never cross: a run of word bytes or of punctuation, each with an
// optional space in front, or a run of whitespace. Python's reference does this with one regex.
std::vector<std::string_view> Pretokenize(std::string_view text) {
  std::vector<std::string_view> chunks;
  size_t i = 0;
  while (i < text.size()) {
    size_t start = i;
    auto c = static_cast<unsigned char>(text[i]);
    bool leading_space = c == ' ' && i + 1 < text.size() && !IsSpace(text[i + 1]);
    if (leading_space) c = static_cast<unsigned char>(text[++i]);
    if (IsWord(c)) {
      while (i < text.size() && IsWord(text[i])) ++i;
    } else if (!IsSpace(c)) {
      while (i < text.size() && !IsWord(text[i]) && !IsSpace(text[i])) ++i;
    } else {
      while (i < text.size() && IsSpace(text[i])) ++i;
    }
    chunks.push_back(text.substr(start, i - start));
  }
  return chunks;
}

uint64_t Key(Pair p) { return (static_cast<uint64_t>(p.first) << 32) | static_cast<uint32_t>(p.second); }

// Replaces each pair (a, b) in ids, left to right, with the token `merged`.
void Merge(std::vector<int>& ids, Pair pair, int merged) {
  size_t out = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i + 1 < ids.size() && ids[i] == pair.first && ids[i + 1] == pair.second) {
      ids[out++] = merged;
      ++i;
    } else {
      ids[out++] = ids[i];
    }
  }
  ids.resize(out);
}

// Learns n_merges merges: each time, the most frequent adjacent pair of tokens becomes a new token.
std::vector<Pair> Train(const std::string& text, int n_merges) {
  // Each distinct chunk once, with how often it occurs, as its bytes.
  std::unordered_map<std::string_view, int64_t> counts;
  for (auto chunk : Pretokenize(text)) ++counts[chunk];
  std::vector<std::vector<int>> words;
  std::vector<int64_t> freq;
  for (auto& [chunk, n] : counts) {
    words.emplace_back(chunk.begin(), chunk.end());
    for (auto& id : words.back()) id = static_cast<unsigned char>(id);
    freq.push_back(n);
  }

  // How often each pair occurs, and which words it occurs in.
  std::unordered_map<uint64_t, int64_t> pair_counts;
  std::unordered_map<uint64_t, std::unordered_set<int>> where;
  auto count = [&](int w, int64_t sign) {
    for (size_t i = 0; i + 1 < words[w].size(); ++i) {
      uint64_t key = Key({words[w][i], words[w][i + 1]});
      pair_counts[key] += sign * freq[w];
      if (sign > 0) where[key].insert(w);
    }
  };
  for (int w = 0; w < static_cast<int>(words.size()); ++w) count(w, +1);

  // A max-heap of (count, pair), where ties go to the smallest pair. Entries go stale as merges take
  // occurrences away, and are checked against pair_counts when they come out.
  auto order = [](const std::pair<int64_t, Pair>& a, const std::pair<int64_t, Pair>& b) {
    return a.first != b.first ? a.first < b.first : a.second > b.second;
  };
  std::priority_queue<std::pair<int64_t, Pair>, std::vector<std::pair<int64_t, Pair>>, decltype(order)> heap(order);
  for (auto& [key, n] : pair_counts) heap.push({n, {static_cast<int>(key >> 32), static_cast<int>(key & 0xffffffff)}});

  std::vector<Pair> merges;
  while (static_cast<int>(merges.size()) < n_merges && !heap.empty()) {
    auto [n, pair] = heap.top();
    heap.pop();
    auto it = pair_counts.find(Key(pair));
    if (it == pair_counts.end() || it->second <= 0) continue;
    // Stale: the pair has lost occurrences since, so it goes back in with the count it has now.
    if (it->second != n) {
      heap.push({it->second, pair});
      continue;
    }
    int merged = 256 + static_cast<int>(merges.size());
    merges.push_back(pair);
    // Only the words that contain the pair change: take their pairs out, merge, and put them back.
    auto touched = where[Key(pair)];
    for (int w : touched) {
      count(w, -1);
      Merge(words[w], pair, merged);
      count(w, +1);
      for (size_t i = 0; i + 1 < words[w].size(); ++i) {
        Pair p{words[w][i], words[w][i + 1]};
        heap.push({pair_counts[Key(p)], p});
      }
    }
    pair_counts.erase(Key(pair));
  }
  return merges;
}

// Encoded chunks, at most `capacity` of them: when it's full, the chunk used longest ago makes room. The list keeps
// them in the order they were used, the most recent at the front, and the map finds a chunk's place in the list,
// so a lookup, a move to the front and making room each take the same time, however full the cache is.
class LruCache {
 public:
  explicit LruCache(size_t capacity) : capacity_(capacity) {}

  // The chunk's ids, or nullptr if it isn't cached. Finding it makes it the most recent.
  const std::vector<int>* Find(std::string_view chunk) {
    auto it = places_.find(chunk);
    if (it == places_.end()) return nullptr;
    entries_.splice(entries_.begin(), entries_, it->second);  // moves the entry, without copying it
    return &it->second->second;
  }

  void Add(std::string_view chunk, std::vector<int> ids) {
    if (capacity_ == 0) return;
    if (entries_.size() == capacity_) {
      places_.erase(entries_.back().first);
      entries_.pop_back();
    }
    entries_.emplace_front(std::string(chunk), std::move(ids));
    // The map's key views the list's own copy of the chunk, which stays put as the list's entries move.
    places_[entries_.front().first] = entries_.begin();
  }

 private:
  using Entries = std::list<std::pair<std::string, std::vector<int>>>;
  size_t capacity_;
  Entries entries_;
  std::unordered_map<std::string_view, Entries::iterator> places_;
};

class Encoder {
 public:
  Encoder(const std::vector<Pair>& merges, const std::map<std::string, int>& specials, size_t cache_size)
      : specials_(specials.begin(), specials.end()), cache_size_(cache_size) {
    for (size_t r = 0; r < merges.size(); ++r) ranks_[Key(merges[r])] = static_cast<int>(r);
  }

  // Each special token's text, such as end of text, becomes its id, and the text between them is encoded on its
  // own, so that no chunk crosses a special token. The cache is the call's own: Encode only reads the encoder, so
  // threads can share one.
  std::vector<int> Encode(std::string_view text) const {
    LruCache cache(cache_size_);
    std::vector<int> ids;
    // Where each special token is next, found again only once we're past it, so the text is searched once.
    std::vector<size_t> next;
    for (auto& [special, id] : specials_) next.push_back(text.find(special));
    size_t start = 0;
    while (true) {
      size_t k = std::min_element(next.begin(), next.end()) - next.begin();
      size_t at = k < next.size() ? next[k] : std::string_view::npos;
      EncodeText(text.substr(start, at == std::string_view::npos ? text.size() - start : at - start), cache, ids);
      if (at == std::string_view::npos) break;
      ids.push_back(specials_[k].second);
      start = at + specials_[k].first.size();
      for (size_t j = 0; j < next.size(); ++j) {
        if (next[j] != std::string_view::npos && next[j] < start) next[j] = text.find(specials_[j].first, start);
      }
    }
    return ids;
  }

 private:
  void EncodeText(std::string_view text, LruCache& cache, std::vector<int>& ids) const {
    for (auto chunk : Pretokenize(text)) {
      const std::vector<int>* cached = cache.Find(chunk);
      if (cached) {
        ids.insert(ids.end(), cached->begin(), cached->end());
      } else {
        std::vector<int> encoded = EncodeChunk(chunk);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
        cache.Add(chunk, std::move(encoded));
      }
    }
  }

  // Applies the merges in the order they were learned: the pair with the lowest rank, until none is left.
  std::vector<int> EncodeChunk(std::string_view chunk) const {
    std::vector<int> ids(chunk.begin(), chunk.end());
    for (auto& id : ids) id = static_cast<unsigned char>(id);
    while (ids.size() > 1) {
      int best = -1;
      Pair best_pair;
      for (size_t i = 0; i + 1 < ids.size(); ++i) {
        auto r = ranks_.find(Key({ids[i], ids[i + 1]}));
        if (r != ranks_.end() && (best < 0 || r->second < best)) {
          best = r->second;
          best_pair = {ids[i], ids[i + 1]};
        }
      }
      if (best < 0) break;
      Merge(ids, best_pair, 256 + best);
    }
    return ids;
  }

  std::vector<std::pair<std::string, int>> specials_;
  std::unordered_map<uint64_t, int> ranks_;
  size_t cache_size_;  // chunks repeat a lot: "Once", " upon", " a"
};

// The bytes malloc has handed out and not had back yet: our C++'s memory, and Python's bigger objects, which come from
// malloc too. A process's memory, as the system counts it, also has memory that was freed and that the allocator
// hasn't given back.
size_t Allocated() {
#ifdef __APPLE__
  malloc_statistics_t stats;
  malloc_zone_statistics(nullptr, &stats);
  return stats.size_in_use;
#else
  struct mallinfo2 info = mallinfo2();
  return info.uordblks + info.hblkhd;
#endif
}

}  // namespace

PYBIND11_MODULE(bpe, m) {
  m.doc() = "Byte-level BPE, in C++";
  m.def("pretokenize", [](py::bytes text) {
    std::string s = text;
    std::vector<py::bytes> out;
    for (auto chunk : Pretokenize(s)) out.emplace_back(std::string(chunk));
    return out;
  });
  m.def("train", [](py::bytes text, int n_merges) {
    std::string s = text;
    py::gil_scoped_release release;
    return Train(s, n_merges);
  });
  m.def("allocated", &Allocated);
  py::class_<Encoder>(m, "Encoder")
      .def(py::init<const std::vector<Pair>&, const std::map<std::string, int>&, size_t>(), py::arg("merges"),
           py::arg("specials") = std::map<std::string, int>(), py::arg("cache_size") = 1 << 16)
      // UTF-8 bytes in, a NumPy array of ids out. The bytes are read where Python keeps them, and as the C++ doesn't
      // touch Python while it encodes, it lets go of the GIL, so that other threads can run Python meanwhile.
      .def("encode", [](const Encoder& e, py::bytes text) {
        std::string_view s = text;
        std::vector<int> ids;
        {
          py::gil_scoped_release release;
          ids = e.Encode(s);
        }
        return py::array_t<int32_t>(ids.size(), ids.data());
      });
}
