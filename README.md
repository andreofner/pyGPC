![alt Overview](https://github.com/andreofner/pyGPC/blob/master/summary_GPC.drawio.png)

## Simultaneous generation and classification with a static model
![alt Overview](https://github.com/andreofner/pyGPC/blob/master/hierarchical_generation_train.png)

A static predictive coding (without dynamical weights and generalized coordinates)
trained to predict pixels. Provided classes replace the deepest state. 
The same network can be used for image synthesis (when labels are provided) or classification (when images are provided).

## Precision inference in static models
![alt Precision](https://github.com/andreofner/pyGPC/blob/master/precision.png)

A closer look at the estimated prediction error precision. Areas that differentiate a certain class (here 7) from other classes
result in higher precision. This distribution of implicit attention across the image is also reflected in predictions coming from
deep layers of the network (darker predictions within high precision regions).

## Dynamical prediction of a Lorenz attractor (ordinary differential equations system)
![alt Overview](https://github.com/andreofner/pyGPC/blob/master/lorenz_attractorFalse.png)
Prediction of hidden states (x1, x2, x3) motion with known cause states (attractor parameters Prandtl & Rayleigh number)

![alt Overview](https://github.com/andreofner/pyGPC/blob/master/lorenz_attractorTrue.png)
Prediction of the same attractor with precision weighted prediction errors

### Have a look at the [demo notebook](https://github.com/andreofner/pyGPC/blob/master/demo.ipynb) for more results.

