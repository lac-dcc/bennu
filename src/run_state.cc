#include "run_state.h"

#include <iostream>

#include "search_policy/sketch_policy.h"

namespace tvm {
namespace auto_scheduler {

Array<tvm::auto_scheduler::MeasureResult> Run(const tvm::auto_scheduler::SearchTask& search_task,
                                              const tvm::auto_scheduler::State& state,
                                              const int timeout = 15, const int number = 3,
                                              const int repeat = 10, const int min_repeat_ms = 100,
                                              const double cooldown_interval = 0.0,
                                              const bool enable_cpu_cache_flush = false,
                                              const int device = 0, const int n_parallel = 12,
                                              const tvm::runtime::String& build_func = "default") {
  auto runner = tvm::auto_scheduler::LocalRunner(timeout, number, repeat, min_repeat_ms,
                                                 cooldown_interval, enable_cpu_cache_flush, device);
  auto builder = tvm::auto_scheduler::LocalBuilder(timeout, n_parallel, build_func);
  Array<tvm::auto_scheduler::MeasureCallback> measure_callbacks;
  auto verbose = 0;

  tvm::auto_scheduler::ProgramMeasurer measurer =
      tvm::auto_scheduler::ProgramMeasurer(builder, runner, measure_callbacks, verbose);

  Array<tvm::auto_scheduler::MeasureInput> inputs;
  Array<tvm::auto_scheduler::MeasureResult> results;

  auto node = tvm::auto_scheduler::SketchPolicyNode();

  measurer->Reset();
  inputs.push_back(tvm::auto_scheduler::MeasureInput(search_task, state));
  results =
      measurer->Measure(search_task, GetRef<tvm::auto_scheduler::SearchPolicy>(&node), inputs);
  return results;
}

TVM_REGISTER_GLOBAL("auto_scheduler.Run").set_body_typed(Run);

}  // namespace auto_scheduler
}  // namespace tvm