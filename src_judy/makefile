# @Author: Xi He <Heerye>
# @Date:   2018-02-22T16:25:10-05:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: makefile
# @Last modified by:   Heerye
# @Last modified time: 2018-03-05T21:48:22-05:00

gpu:
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true -lr 0.1 -b 1200 -N 2000
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true -lr 0.05 -b 1200 -N 2000
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true -lr 0.01 -b 1200 -N 2000

cpu:
		python3 main.py -gpu false -lr 0.1 -N 1000

batch:
	# CUDA_VISIBLE_DEVICES=0 python3 -O main.py -gpu true -lr 0.01 -b 1000 -N 20000 -reuse_params false -resume false
	# CUDA_VISIBLE_DEVICES=0 python3 main.py -gpu true -lr 0.001 -b 1000 -N 10000 -reuse_params true -resume true
	# cp ./checkpoints/3_0.pt ./checkpoints/3_0_1000_3.pt
	# CUDA_VISIBLE_DEVICES=0 python3 -O main.py -gpu true -lr 0.01 -b 500 -N 5000 -reuse_params true -resume true
	# CUDA_VISIBLE_DEVICES=0 python3 main.py -gpu true -lr 0.01 -b 32 -N 10000 -reuse_params true -resume true
	# cp ./checkpoints/3_0.pt ./checkpoints/3_0_32.pt

second:
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true  -b 200 -N 10000
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true  -b 500 -N 10000
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true  -b 1000 -N 10000
	CUDA_VISIBLE_DEVICES=3 python3 -O main.py -gpu true  -b 2000 -N 10000
