#!/bin/sh
# Runs a command with the bpe module built with ThreadSanitizer, which reports two threads touching the same memory
# without anything ordering them, then builds the module as usual again:
#   ./tsan.sh python bpe_measure.py threads --pieces 2
SANITIZE=thread python setup.py build_ext --inplace --force > /dev/null 2>&1
# Python isn't built with ThreadSanitizer, so its runtime is loaded into the process first.
if [ "$(uname)" = Darwin ]; then
  runtime="DYLD_INSERT_LIBRARIES=$(clang -print-file-name=libclang_rt.tsan_osx_dynamic.dylib)"
else
  runtime="LD_PRELOAD=$(gcc -print-file-name=libtsan.so)"
fi
# It stops at the first report, and each frame is its file and line, then its function.
env "$runtime" TSAN_OPTIONS="halt_on_error=1 stack_trace_format='    #%n %S %f'" "$@"
status=$?
python setup.py build_ext --inplace --force > /dev/null 2>&1
exit $status
