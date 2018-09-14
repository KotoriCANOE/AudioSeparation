# Note

## 01

kernel0 = [1, 8]
kernel1 = [1, 4]
kernel2 = [1, 3]
channels: 32, 48, 64, 96, 128
ResBlocks: 0, 0, 1, 1, 2 | 1, 1, 0, 0, 0
use dense connection

batch-size: 1

## 02

kernel0 = [1, 8]
kernel1 = [1, 4]
kernel2 = [1, 3]
channels: 32, 48, 64, 96, 128, 160, 192, 224, 256
ResBlocks: 0, 0, 1, 1, 2, 2, 2, 3, 3 | 3, 2, 2, 2, 1, 1, 0, 0, 0
use dense connection

batch-size: 1

## 03

added SEUnit to EBlock and DBlock

batch-size: 2

## 04

added normalizer to EBlock and DBlock

batch-size: 1

## 05

same as ## 02
fixed Variable EMA update

batch-size: 2
