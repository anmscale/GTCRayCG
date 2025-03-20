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
NUM_LEAD_MICROBATCHES = 4

with ray.dag.InputNode() as inp:
    fwd_queues = [[] for _ in range(NUM_WORKERS)]
    bwd_queues = [[] for _ in range(NUM_WORKERS)]
    # Once a worker's counter reaches 0, it cannot execute another fwd until it
    # executes a bwd first.
    fwd_counter = [NUM_LEAD_MICROBATCHES - i for i in range(NUM_WORKERS)]
    # All of the done batches.
    done = []

    # FWD on worker 0.
    for i in range(NUM_MICROBATCHES):
        fwd_queues[0].append(inp[i])

    while len(done) < NUM_MICROBATCHES:
        for i, worker in enumerate(workers):
            if fwd_counter[i] > 0 and fwd_queues[i]:
                b = fwd_queues[i].pop(0)
                b = worker.fwd.bind(b)
                if i < NUM_WORKERS - 1:
                    fwd_queues[i + 1].append(b)
                else:
                    bwd_queues[i].append(b)
                fwd_counter[i] -= 1
            elif bwd_queues[i]:
                b = bwd_queues[i].pop(0)
                b = worker.bwd.bind(b)
                if i > 0:
                    bwd_queues[i - 1].append(b)
                else:
                    done.append(b)
                fwd_counter[i] += 1

    dag = ray.dag.MultiOutputNode(done)


dag = dag.experimental_compile()
ray.get(dag.execute(*range(NUM_MICROBATCHES)))
print(ray.get(workers[0].pop_trace.remote()))