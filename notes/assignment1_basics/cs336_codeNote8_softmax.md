# softmax

Write a function to apply the softmax operation on a tensor.

Your function should take two parameters:

a tensor and a dimension i.

and apply softmax to the i-th dimension of the input tensor.

# 代码实现

softmax 没有参数，是比较简单的一个

```py
class Softmax(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        i: int
    ):
        max_value = x.max(dim = i , keepdim=True).values
        
        x_exp = torch.exp(x-max_value)
        partition = x_exp.sum(i, keepdim= True)
        
        return x_exp/partition
```

这边我们用的是稳定的softmax实现，先减去最大值再做exp，避免溢出。

因为我们就算给 x_exp 的指数全部减去一个常数 c，softmax 的结果也不会变，因为分子和分母都会乘上 exp(-c)，所以我们可以减去最大值来避免溢出。

第一次做的时候因为没有减去最大值，导致在测试的时候出现了溢出，结果都是 nan。


测试均通过