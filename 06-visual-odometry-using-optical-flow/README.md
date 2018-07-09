# Visual Odometry Using Direct Methods

---

## Lukas-Kanade Optical Flow

### 1. Research Review

#### a. How many categories could Lukas-Kanade algorithm be classified into?

**Four** categories. They are forward additive, forward compositional, inverse additive and inverse compositional.

#### b. Why warp of original image is needed for compositional methods? What is its physical meaning?

#### c. What are the differences between forward and inverse methods?

### 2. Forward Additive through Gauss-Newton

### 3. Inverse Additive through Gauss-Newton

### 4. Coarse-to-Fine Pyramid

<img src="doc/01-a-single-level--forward.jpg" alt="Single Level, Forward" width="100%">

<img src="doc/01-b-single-level--inverse.jpg" alt="Single Level, Inverse" width="100%">

<img src="doc/01-c-multi-level--forward.jpg" alt="Multi Level, Forward" width="100%">

<img src="doc/01-d-multi-level--inverse.jpg" alt="Multi Level, Inverse" width="100%">

<img src="doc/01-e-opencv.jpg" alt="OpenCV Reference" width="100%">

## Direct Method

### 1. Pose Estimation Results:

#### a. Single Layer Method:

000001.png
```shell
T21 =
   0.999991  0.00241676  0.00338662 -0.00235501
-0.00242396    0.999995   0.0021214  0.00396548
-0.00338148 -0.00212959    0.999992    -0.72423
          0           0           0           1
```

000002.png
```shell
T21 =
    0.999972   0.00167569   0.00731382   0.00720773
 -0.00170428     0.999991   0.00390376 -0.000293629
 -0.00730722  -0.00391612     0.999966     -1.46784
           0            0            0            1
```

000003.png
```shell
T21 =
   0.999904  0.00157543    0.013745    -0.24652
-0.00164949    0.999984  0.00537809  0.00134025
 -0.0137364 -0.00540025    0.999891    -1.85275
          0           0           0           1
```

000004.png
```shell
T21 =
    0.99985  0.00315904    0.017035   -0.308368
-0.00323999    0.999984  0.00472606   0.0292316
 -0.0170198 -0.00478054    0.999844    -2.03269
          0           0           0           1
```

000005.png
```shell
T21 =
   0.999778  0.00390722    0.020702   -0.421147
-0.00401972    0.999977  0.00539539   0.0411062
 -0.0206805 -0.00547741    0.999771    -2.26976
          0           0           0           1
```

#### b. Multi Layer Method:

000001.png
```shell
T21 =
   0.999991  0.00241676  0.00338662 -0.00235499
-0.00242395    0.999995   0.0021214  0.00396545
-0.00338148 -0.00212959    0.999992    -0.72423
          0           0           0           1
```

000002.png
```shell
T21 =
    0.999972   0.00167569   0.00731382   0.00720773
 -0.00170428     0.999991   0.00390376 -0.000293629
 -0.00730722  -0.00391612     0.999966     -1.46784
           0            0            0            1
```

000003.png
```shell
T21 =
   0.999937  0.00185644   0.0110728   0.0110558
-0.00191404    0.999985  0.00519405  0.00118652
  -0.011063 -0.00521492    0.999925    -2.21266
          0           0           0           1
```

000004.png
```shell
T21 =
    0.999874  0.000272158    0.0158774    0.0092791
-0.000362705     0.999984   0.00570031   0.00328783
  -0.0158756  -0.00570535     0.999858     -2.99918
           0            0            0            1
```

000005.png
```shell
T21 =
   0.999803   0.0011951   0.0198032   0.0193427
-0.00132664    0.999977  0.00663092  -0.0109045
 -0.0197949 -0.00665588    0.999782    -3.79221
          0           0           0           1
```
