In this project, we recreate the temporal prediction example presented in,

Elsayed, M., Vasan, G., & Mahmood, A. R. (2024). Streaming Deep Reinforcement Learning Finally Works. arXiv preprint arXiv:2410.14606.

This example is built on the electricity transformer dataset (https://github.com/zhouhaoyi/ETDataset), which is described in

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021, May). Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 12, pp. 11106-11115).

The dataset contains seven time-series signals, and our prediction target will be one of them -- the oil temperature.
At a given time $t$, we will use the present value of the six other variables and the previous ($t-1$) value of the oil temperature to predict future oil temperatures.

Rather than predicting the raw temperature at time $t+1$, we predict the ``return'', which is an aggregation of discounted future oil temperatures:

$$ G_t = \sum_{k=t+1}^{\infty} \gamma^{k-t-1} R_k $$.

In the standard RL setting, $R_k$ refers to reward. Here, we will treat this as the oil temperature at time $k$.
Essentially, we are now in the realm of Generalized Value Functions (GVF), and our goal is to estimate this value function.
This task falls into the category of online learning -- the incoming data is acts as a stream, and we update our value/prediction function in real-time.

To do so, we will train a neural network following the TD learning scheme with the modifications suggested in our primary reference.

The above example can be run by entering ```python run_example.py``` in the terminal. This code generates two plots.
The first, below, illustrates the variables in the dataset.

![io_signals](https://github.com/user-attachments/assets/d2fd7297-022b-43b0-ac37-511d8dd502b2)

The second, below, illustrates the predicted ``return'' of our learned GVF at each time step compared to the actual return.

![return_predictions](https://github.com/user-attachments/assets/952506fc-de95-4c8e-a3ae-208db5e1dcbb)

It is evident that the learned model adapts and improves over time.

There is naturally quite a bit of literature in this space. There are plenty of other resources that provide a more comprehensive listing, but here are a few papers that I recently read and learned from:

Janjua, M. K., Shah, H., White, M., Miahi, E., Machado, M. C., & White, A. (2024). GVFs in the real world: making predictions online for water treatment. Machine Learning, 113(8), 5151-5181.

Ring, M. (2021). Representing knowledge as predictions (and state as knowledge). arXiv preprint arXiv:2112.06336.

Lyle, C., Zheng, Z., Khetarpal, K., Martens, J., van Hasselt, H. P., Pascanu, R., & Dabney, W. (2024). Normalization and effective learning rates in reinforcement learning. Advances in Neural Information Processing Systems, 37, 106440-106473.
