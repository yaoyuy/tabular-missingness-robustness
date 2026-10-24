# Vendored data

Local copies of the two UCI files used in the main text, because live UCI
links can change (the Pima UCI link already returned 404 during this work).

- `ionosphere.data`
  Source: https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
  SHA-256: 46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd
  Shape: 351 rows, 35 columns; last column 'g' 225, 'b' 126.

- `processed.cleveland.data`
  Source: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
  SHA-256: a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8
  Rows with '?' are dropped by the scripts (303 -> 297), then the target is
  binarised as 0 versus >0 (160 versus 137).

Note: the scripts in this repository still download from UCI at run time
by default. These local copies are provided so a rerun does not depend on
a live link; to use them, point the download step at these files or
compare a fresh download's SHA-256 with the hashes above.

Datasets remain under their original terms.
