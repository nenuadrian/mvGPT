// Byte-level BPE: training merges on a text, and encoding text with them. Exposed to Python with pybind11.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
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

class Encoder {
 public:
  explicit Encoder(const std::vector<Pair>& merges) {
    for (size_t r = 0; r < merges.size(); ++r) ranks_[Key(merges[r])] = static_cast<int>(r);
  }

  std::vector<int> Encode(const std::string& text) {
    std::vector<int> ids;
    for (auto chunk : Pretokenize(text)) {
      auto cached = cache_.find(std::string(chunk));
      if (cached == cache_.end()) cached = cache_.emplace(std::string(chunk), EncodeChunk(chunk)).first;
      ids.insert(ids.end(), cached->second.begin(), cached->second.end());
    }
    return ids;
  }

 private:
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

  std::unordered_map<uint64_t, int> ranks_;
  std::unordered_map<std::string, std::vector<int>> cache_;  // chunks repeat a lot: "Once", " upon", " a"
};

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
  py::class_<Encoder>(m, "Encoder")
      .def(py::init<const std::vector<Pair>&>())
      .def("encode", [](Encoder& e, py::bytes text) {
        std::string s = text;
        return e.Encode(s);
      });
}
