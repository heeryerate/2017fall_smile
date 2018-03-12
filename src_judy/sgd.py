############

#   @File name: sgd.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-03-04 11:52:28

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-03-04 11:53:04

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############


def sgd():
    grad = get_grad(net, x, y)
    for j, para in enumerate(get_params(net)):
        para.data -= args.lr * grad[j].data
