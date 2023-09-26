import matplotlib.pyplot as plt
import os

o = """/usr/local/bin/python3.11 /Users/royzhao/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Coding/Face_Detect/eye_feature_points_detection.py 
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/torch/nn/modules/lazy.py:180: UserWarning: Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.
  warnings.warn('Lazy modules are a new feature under heavy development '
Epoch 335
100%|██████████| 7/7 [00:47<00:00,  6.83s/it, Training Loss==5.51e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.98it/s, Validation Loss==8.3e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 336
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.73e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.64it/s, Validation Loss==0.000105]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 337
100%|██████████| 7/7 [00:48<00:00,  6.90s/it, Training Loss==5.51e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.97it/s, Validation Loss==8.47e-5]
Epoch 338
100%|██████████| 7/7 [00:48<00:00,  6.86s/it, Training Loss==5.37e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.10it/s, Validation Loss==9.09e-5]
Epoch 339
100%|██████████| 7/7 [00:48<00:00,  7.00s/it, Training Loss==5.52e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.88it/s, Validation Loss==8.69e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 340
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.54e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.15it/s, Validation Loss==9.05e-5]
Epoch 341
100%|██████████| 7/7 [00:49<00:00,  7.05s/it, Training Loss==5.88e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.25it/s, Validation Loss==8.79e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 342
100%|██████████| 7/7 [00:48<00:00,  6.97s/it, Training Loss==5.58e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.10it/s, Validation Loss==8.87e-5]
Epoch 343
100%|██████████| 7/7 [00:48<00:00,  6.88s/it, Training Loss==5.8e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.23it/s, Validation Loss==8.95e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 344
100%|██████████| 7/7 [00:47<00:00,  6.73s/it, Training Loss==5.62e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.47it/s, Validation Loss==8.64e-5]
Epoch 345
100%|██████████| 7/7 [00:46<00:00,  6.71s/it, Training Loss==5.54e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.93it/s, Validation Loss==8.59e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 346
100%|██████████| 7/7 [00:47<00:00,  6.76s/it, Training Loss==5.78e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 11.00it/s, Validation Loss==9.21e-5]
Epoch 347
100%|██████████| 7/7 [00:47<00:00,  6.73s/it, Training Loss==6.32e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.68it/s, Validation Loss==0.000106]
Epoch 348
100%|██████████| 7/7 [00:46<00:00,  6.63s/it, Training Loss==5.76e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.61it/s, Validation Loss==8.41e-5]
Epoch 349
100%|██████████| 7/7 [00:47<00:00,  6.74s/it, Training Loss==5.51e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.35it/s, Validation Loss==9.64e-5]
Epoch 350
100%|██████████| 7/7 [00:46<00:00,  6.71s/it, Training Loss==5.8e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.30it/s, Validation Loss==8.25e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 351
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==6.06e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.95it/s, Validation Loss==8.84e-5]
Epoch 352
100%|██████████| 7/7 [00:46<00:00,  6.70s/it, Training Loss==5.98e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.96it/s, Validation Loss==8.94e-5]
Epoch 353
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.77e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.41it/s, Validation Loss==9.13e-5]
Epoch 354
100%|██████████| 7/7 [00:47<00:00,  6.81s/it, Training Loss==5.89e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.46it/s, Validation Loss==8.67e-5]
Epoch 355
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.4e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.11it/s, Validation Loss==0.000106]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 356
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==6.19e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.09it/s, Validation Loss==8.97e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 357
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==5.84e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00,  9.97it/s, Validation Loss==8.81e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 358
100%|██████████| 7/7 [00:47<00:00,  6.78s/it, Training Loss==5.4e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.43it/s, Validation Loss==9.25e-5]
Epoch 359
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==5.72e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.32it/s, Validation Loss==0.000105]
Epoch 360
100%|██████████| 7/7 [00:47<00:00,  6.73s/it, Training Loss==5.74e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.52it/s, Validation Loss==8.74e-5]
Epoch 361
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.61e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.77it/s, Validation Loss==0.000107]
Epoch 362
100%|██████████| 7/7 [00:47<00:00,  6.76s/it, Training Loss==5.76e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.66it/s, Validation Loss==8.83e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 363
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==6.05e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.52it/s, Validation Loss==8.66e-5]
Epoch 364
100%|██████████| 7/7 [00:47<00:00,  6.78s/it, Training Loss==5.92e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.49it/s, Validation Loss==8.67e-5]
Epoch 365
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.86e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.35it/s, Validation Loss==8.66e-5]
Epoch 366
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.99e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.73it/s, Validation Loss==8.76e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 367
100%|██████████| 7/7 [00:47<00:00,  6.78s/it, Training Loss==5.75e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.29it/s, Validation Loss==9.11e-5]
Epoch 368
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.53e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.82it/s, Validation Loss==0.000107]
Epoch 369
100%|██████████| 7/7 [00:47<00:00,  6.81s/it, Training Loss==5.59e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.08it/s, Validation Loss==0.000113]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 370
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==6.19e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.59it/s, Validation Loss==9.09e-5]
Epoch 371
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.89e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.50it/s, Validation Loss==8.95e-5]
Epoch 372
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==5.68e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00,  9.24it/s, Validation Loss==0.000107]
Epoch 373
100%|██████████| 7/7 [00:47<00:00,  6.77s/it, Training Loss==5.72e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00,  9.80it/s, Validation Loss==8.72e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 374
100%|██████████| 7/7 [00:46<00:00,  6.68s/it, Training Loss==5.73e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 11.06it/s, Validation Loss==8.29e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 375
100%|██████████| 7/7 [00:46<00:00,  6.62s/it, Training Loss==5.83e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.79it/s, Validation Loss==8.26e-5]
Epoch 376
100%|██████████| 7/7 [00:47<00:00,  6.76s/it, Training Loss==5.68e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.55it/s, Validation Loss==8.52e-5]
Epoch 377
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.4e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.52it/s, Validation Loss==0.000104]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 378
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==5.57e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.07it/s, Validation Loss==0.000101]
Epoch 379
100%|██████████| 7/7 [00:47<00:00,  6.76s/it, Training Loss==5.83e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00,  9.96it/s, Validation Loss==0.000104]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 380
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.76e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.61it/s, Validation Loss==8.8e-5]
Epoch 381
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.78e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.22it/s, Validation Loss==0.000109]
Epoch 382
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.79e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.57it/s, Validation Loss==0.000106]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 383
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==5.63e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.59it/s, Validation Loss==8.73e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 384
100%|██████████| 7/7 [00:46<00:00,  6.65s/it, Training Loss==5.92e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.83it/s, Validation Loss==0.000105]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 385
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==5.78e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.28it/s, Validation Loss==8.85e-5]
Epoch 386
100%|██████████| 7/7 [00:47<00:00,  6.78s/it, Training Loss==5.55e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.11it/s, Validation Loss==8.36e-5]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 387
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==6e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.91it/s, Validation Loss==8.87e-5]
Epoch 388
100%|██████████| 7/7 [00:47<00:00,  6.79s/it, Training Loss==5.31e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.19it/s, Validation Loss==8.72e-5]
Epoch 389
100%|██████████| 7/7 [00:47<00:00,  6.74s/it, Training Loss==5.53e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.51it/s, Validation Loss==0.000101]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 390
100%|██████████| 7/7 [00:47<00:00,  6.71s/it, Training Loss==5.99e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.35it/s, Validation Loss==0.000107]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 391
100%|██████████| 7/7 [00:47<00:00,  6.75s/it, Training Loss==6.05e-5]
  0%|          | 0/46 [00:00<?, ?it/s]Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.47it/s, Validation Loss==0.000113]
  0%|          | 0/7 [00:00<?, ?it/s]Epoch 392
100%|██████████| 7/7 [00:47<00:00,  6.80s/it, Training Loss==5.76e-5]
Evaluation
100%|██████████| 46/46 [00:04<00:00, 10.38it/s, Validation Loss==8.67e-5]
Epoch 393
 14%|█▍        | 1/7 [00:07<00:44,  7.42s/it, Training Loss==5.41e-5]"""

prefix = "4_points_resnet34_linear_1_small_img_"
train = []
val = []

if len(o) > 0:
    o = o.split('\n')

    i = []
    for l in range(len(o)):
        try:
            if o[l][-1] == ']':
                i.append(o[l])
        except:
            pass

    for l in range(len(i)):
        if l % 2 == 0:
            train.append(i[l])
        else:
            val.append(i[l])

    train = [float(train[i].split("Loss==")[1].split("]")[0]) for i in range(len(train))]
    val = [float(val[i].split("Loss==")[1].split("]")[0]) for i in range(len(val))]
    plt.semilogy(train)
    plt.semilogy(val)
    plt.xscale('log')


else:
    o = os.listdir("weights")
    x = []
    y = []
    for item in o:
        if item.split("epoch=")[0] == prefix:
            x.append(float(item.split("epoch=")[1].split("_loss")[0]))
            y.append(float(item.split("epoch=")[1].split("loss=")[1].split(".pth")[0]))

    plt.scatter(x, y)
    plt.yscale('log')
    plt.xscale('log')

plt.show()
