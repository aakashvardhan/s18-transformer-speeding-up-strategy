# Speeding up the training of the Transformer model with optimization techniques

## Table of Contents
1. [Introduction](#introduction)
2. [Key Improvements](#key-improvements)
3. [Results](#results)
4. [Conclusion](#conclusion)

## Introduction

The Transformation model is known for its parallelization capabilities and its ability to capture long-range dependencies in sequences. However, training the Transformer model can be computationally expensive due to the large number of parameters and the complexity of the model architecture. In this project, we explore various optimization techniques to speed up the training of the Transformer model. We experiment with various optimization techniques to reduce the training time of the model. We evaluate the performance of these techniques on a machine translation task using the opus-book English to Italian dataset.

## Key Improvements

We experiment with the following optimization techniques to speed up the training of the Transformer model:

- **16-bit Mixed Precision Training**: We use mixed precision training to speed up the training of the model by reducing the memory footprint and increasing the computation speed. Using Pytorch lightning, we can easily enable mixed precision training by setting the `precision=16-mixed` flag in the `Trainer` class.

- **Gradient Clipping**: We use gradient clipping to prevent the exploding gradient problem during training. We set the maximum gradient norm to 0.5 to prevent the gradients from becoming too large. Using Pytorch lightning, we can easily enable gradient clipping by setting the `gradient_clip_val=0.5` flag in the `Trainer` class.

- **OneCycle LR Scheduler**: The OneCycleLR scheduler is a learning rate policy designed to improve the training efficiency and performance of neural networks, including transformer models. This technique was proposed in the paper "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates". This approach dynamically adjusts the learning rate during training to speed up the convergence of the model. We use the OneCycleLR scheduler with a maximum learning rate of 5e-4 and a total number of training steps of the length of the `train_dataloader`.

- **Stacking Multiple Encoder and Decoder Layers**: We experiment with stacking multiple encoder and decoder layers to increase the model's capacity and improve its performance. We increase the number of encoder and decoder layers by 6 to enhance the model's ability to capture complex patterns and improve the translation quality.

- **Dynamic Batch Padding**: This is a crucial component for training a transformer model on tasks with variable-length sequences. We use dynamic batch padding to ensure the input sequences are padded to the same length within each batch. This technique helps speed up the training process by reducing the number of computations required to process each batch.

## Results

We evaluate the performance of the Transformer model with the optimization techniques on the opus-book English to Italian dataset. We train the model for 1 epochs and evaluate the translation quality using the BLEU score metric. The results show that the optimization techniques significantly speed up the training process and improve the translation quality of the model.

Final Training and Validation Results:

```
  | Name    | Type             | Params | Mode 
-----------------------------------------------------
0 | model   | Transformer      | 34.9 M | train
1 | loss_fn | CrossEntropyLoss | 0      | train
-----------------------------------------------------
34.9 M    Trainable params
0         Non-trainable params
34.9 M    Total params
139.716   Total estimated model params size (MB)
Max length of the source sentence : 43
Max length of the source target : 38

Epoch 16: 100%
 241/241 [00:33<00:00,  7.20it/s, v_num=1, train_loss_step=1.580, train_loss_epoch=1.580]
```

- With Pytorch Lightning's Early Stopping Callback, the training process was stopped after 16 epochs as the validation loss reached the 1.5 threshold.


16th Epoch Validation Result:

```
--------------------------------------------------------------------------------
    SOURCE: And he loves me; All this has been, but will pass,' she said, feeling that tears of joy at this return to life were running down her cheeks.
    TARGET: Perché lui mi ama! Questo è stato e passerà” ella diceva, sentendo che le lacrime della gioia del ritorno alla vita le scorrevano per le guance.
 PREDICTED: E mi ama , anche così bene , ma sarà quel sentimento di cui le lacrime agli occhi .
--------------------------------------------------------------------------------
    SOURCE: "Mr. Rivers! you quite put me out of patience: I am rational enough; it is you who misunderstand, or rather who affect to misunderstand."
    TARGET: — Signor Rivers, mi farete perder la pazienza; sono calma; siete voi che non capite, o che fingete di non capirmi.
 PREDICTED: — Il signor Rivers era suo aiuto , che ora mi è ora di voi ; ma voi chi è a sbagliare e voi al mio .
--------------------------------------------------------------------------------
    SOURCE: 'What is it?' he asked drily. 'We are busy.'
    TARGET: — Che vuoi? — le disse asciutto. — Siamo occupati.
 PREDICTED: — Che vuoi ? — le disse asciutto . — Siamo .
```


<table>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s18-transformer-speeding-up-strategy/blob/main/asset/train_loss.png" alt="Plot 1" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s18-transformer-speeding-up-strategy/blob/main/asset/val_cer.png" alt="Plot 2" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s18-transformer-speeding-up-strategy/blob/main/asset/val_wer.png" alt="Plot 3" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">Training Loss</td>
    <td align="center">Validation Character Error Rate</td>
    <td align="center">Validation Word Error Rate</td>
  </tr>
</table>

## Conclusion

In this project, we explored various optimization techniques to speed up the training of the Transformer model. We experimented with mixed precision training, gradient clipping, OneCycle LR scheduler, stacking multiple encoder and decoder layers, and dynamic batch padding. The results show that these optimization techniques significantly speed up the training process and improve the translation quality of the model. We achieved a Validation Character Error Rate of 0.1279 and a Validation Word Error Rate of 0.411 after 16 epochs of training. The model's performance can be further improved by fine-tuning the hyperparameters and increasing the training duration. Overall, the optimization techniques presented in this project can be used to speed up the training of the Transformer model and improve its performance on various natural language processing tasks.