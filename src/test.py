# @Author: Xi He <Esparami>
# @Date:   2017-09-07T12:35:54-04:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: test.py
# @Last modified by:   Esparami
# @Last modified time: 2017-09-19T14:23:26-04:00


import matplotlib.pyplot as plt

def foo(this):
    print(this)

if __name__ == '__main__':
    list_data = [1, 2, 3, 4]
    plt.plot(list_data)
    foo(list_data)
