import ray
import ray.dag


@ray.remote
class Worker:
    def __init__(self, rank):
        self.rank = rank
        self.trace = []

    def fwd(self, b):
        print("fwd", self.rank, b)
        self.trace.append(("fwd", b))
        return b

    def bwd(self, b):
        print("bwd", self.rank, b)
        self.trace.append(("bwd", b))
        return b

    def pop_trace(self):
        trace = self.trace
        self.trace = []
        return trace


NUM_WORKERS = 4
workers = [Worker.remote(i) for i in range(NUM_WORKERS)]
NUM_MICROBATCHES = 8

with ray.dag.InputNode() as inp:
    batches = [inp[i] for i in range(NUM_MICROBATCHES)]
    for worker in workers:
        batches = [worker.fwd.bind(batch) for batch in batches]
    for worker in reversed(workers):
        batches = [worker.bwd.bind(batch) for batch in batches]
    dag = ray.dag.MultiOutputNode(batches)


dag = dag.experimental_compile()
ray.get(dag.execute(*range(NUM_MICROBATCHES)))
print(ray.get(workers[0].pop_trace.remote()))