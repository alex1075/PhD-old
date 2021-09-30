from ai_benchmark import AIBenchmark

#Reu benchmark for tensorflow 
benchmark = AIBenchmark(use_CPU=None, verbose_level=3)
# results = benchmark.run(precision="high")
result = benchmark.run()