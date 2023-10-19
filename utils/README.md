python print_record_info.py --logfile '../results/cpu_matmul.json' --index 2

The file ../results/cpu_matmul.json exists.
Index: 2
Time cost (second): [T.float64(0.022992200000000001), T.float64(0.023908800000000001), T.float64(0.0230355)]
----------
Workload Key: ["ansor_mm", 1000, 800, 700, "float32"]
FLOP Ct: 1120000000.0
----------
Compute DAG:
A = PLACEHOLDER [1000, 800]
B = PLACEHOLDER [800, 700]
C(i, j) += (A[i, k]*B[k, j])

Program:
Placeholder: A, B
C auto_unroll: 16
parallel i.0@j.0@i.1@ (0,350)
  for j.1 (0,4)
    for k.0 (0,400)
      for i.2 (0,100)
        for j.2 (0,5)
          for k.1 (0,2)
            C = ...


python print_record_info.py --logfile '../results/cpu_matmul.json' --index 4
The file ../results/cpu_matmul.json exists.
Index: 4
Time cost (second): [T.float64(0.0050583499999999997), T.float64(0.0053649700000000002), T.float64(0.0049591399999999999)]
----------
Workload Key: ["ansor_mm", 1000, 800, 700, "float32"]
FLOP Ct: 1120000000.0
----------
Compute DAG:
A = PLACEHOLDER [1000, 800]
B = PLACEHOLDER [800, 700]
C(i, j) += (A[i, k]*B[k, j])

Program:
Placeholder: A, B
parallel i.0@j.0@i.1@j.1@ (0,100)
  C.local auto_unroll: 64
  for k.0 (0,100)
    for i_c.2 (0,5)
      for j_c.2 (0,35)
        for k.1 (0,8)
          for i_c.3 (0,2)
            for j_c.3 (0,20)
              C.local = ...
  for i.2 (0,10)
    for j.2 (0,700)
      C = ...


