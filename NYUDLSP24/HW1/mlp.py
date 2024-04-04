import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        self.cache['x'] = x

        s1 = torch.matmul(x,self.parameters["W1"].T) + self.parameters["b1"]
        self.cache['s1']=s1
        
        if self.f_function=="relu": a1 = torch.relu(s1)
        elif self.f_function=="sigmoid": a1 = torch.sigmoid(s1)
        else: a1 = s1
        self.cache['a1']=a1
        
        s2 = torch.matmul(a1,self.parameters["W2"].T) + self.parameters["b2"]
        self.cache['s2']=s2
        
        if self.g_function=="relu": y_hat = torch.relu(s2)
        elif self.g_function=="sigmoid": y_hat = torch.sigmoid(s2)
        else: y_hat = s2
        self.cache['y_hat']=y_hat

        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        x=self.cache['x']
        s1=self.cache['s1']
        a1=self.cache['a1']
        s2=self.cache['s2']
        y_hat=self.cache['y_hat']

        # dJ/ds2 = dJ/dy_hat * dy_hat/ds2
        if self.g_function=="relu":
            dJds2 = torch.where(s2>0, dJdy_hat, 0)

        elif self.g_function=="sigmoid":
            dJds2 = dJdy_hat * y_hat * (1-y_hat)
 
        else:
            dJds2 = dJdy_hat

        # dJ/da1 = dJ/ds2 * ds2/da1
        dJda1 = torch.matmul(dJds2,self.parameters['W2'])

        # dJ/db2 = dJ/ds2 * ds2/db2
        dJdb2 = dJds2.sum(0)
        self.grads['dJdb2'] = dJdb2

        # dJ/dW2 = dJ/ds2 * ds2/dW2
        dJdW2 = torch.matmul(dJds2.T,a1)
        self.grads['dJdW2'] = dJdW2
    
        # dJ/ds1 = dJ/da1 * da1/ds1
        if self.f_function=="relu":
            dJds1 = torch.where(s1>0, dJda1, 0)

        elif self.f_function=="sigmoid":
            dJds1 = dJda1 * a1 * (1-a1)
 
        else:
            dJds1 = dJda1

        # dJ/db1 = dJ/ds1 * ds1/db1
        dJdb1 = dJds1.sum(0)
        self.grads['dJdb1'] = dJdb1

        # dJ/dW1 = dJ/ds1 * ds1/dW1
        dJdW1 = torch.matmul(dJds1.T,x)
        self.grads['dJdW1'] = dJdW1

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    return torch.mean((y_hat - y) ** 2), 2 * (y_hat - y)/(y.shape[0] * y.shape[1])

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    return - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)).mean(), (y_hat - y) / (y_hat * (1 - y_hat) * y.shape[0] * y.shape[1])











