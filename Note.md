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

(unchanged)
added SEUnit to EBlock and DBlock

batch-size: 2

## 04

added normalizer to EBlock and DBlock

batch-size: 1

## 05

same as ## 02
fixed Variable EMA update

batch-size: 2

## 06

(unchanged)
same as ## 02
no SEUnit
added normalizer

batch-size: 2

## 07

kernel1 = [1, 4] => [1, 8]

## 08

kernel2 = [1, 3] => [1, 5]
SGDR - m_mul: 1.0 => 0.85

## 09

kernel2 = [1, 7]

## 10

kernel2 = [1, 3]

## 11

(unchanged)
added normalizer to EBlock and DBlock

## 12

(unchanged)
InBlock: [1, 8] => [1, 16]

## 13

added EBlock without down/up-sampling at the beginning/ending
channels: 16, 32, 48, 64, 96, 128, 160, 192, 224, 256
ResBlocks: 0, 0, 0, 1, 1, 2, 2, 2, 3, 3 | 3, 2, 2, 2, 1, 1, 0, 0, 0, 0

## 14

use residual connection

## 15

loss: MS-SSIM

## 16

loss: L1 + MS-SSIM

## 20

steps: 127000
dataset: expanded to 16 songs
loss: L1
batch size: 2

## 21

steps: 127000
packed data
dataset: added random amplitude
batch size: 4

