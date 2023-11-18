#ifndef RUN_STATE_H
#define RUN_STATE_H

#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/auto_schedule.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace auto_schedule {

Array<tvm::auto_scheduler::MeasureResult> Run(const tvm::auto_scheduler::SearchTask& search_task,
                                              const tvm::auto_scheduler::State& state,
                                              const int timeout = 15, const int number = 3,
                                              const int repeat = 10, const int min_repeat_ms = 100,
                                              const double cooldown_interval = 0.0,
                                              const bool enable_cpu_cache_flush = false,
                                              const int device = 0, const int n_parallel = 12,
                                              const tvm::runtime::String& build_func = "default");

} // namespace auto_schedule
} // namespace tvm

#endif