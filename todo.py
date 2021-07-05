from model import UNetwork
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os import listdir
from tool import *
import scipy.misc
import h5py

def train(args):

    f_edge = h5py.File('DATA_THYROID_SAMPLE.mat')
    eg = f_edge['data']

    RF_all = eg['RF']
    RF_all = np.transpose(RF_all)

    label_all = eg['atten']
    label_all = np.transpose(label_all)
    label_AT = np.array(label_all, np.float32) / 255

    label_all = eg['sos']
    label_all = np.transpose(label_all)
    label_SOS = np.array(label_all, np.float32) / 255

    label_all = eg['scatter_radi']
    label_all = np.transpose(label_all)
    label_SC1 = np.array(label_all, np.float32) / 255

    label_all = eg['scatter_mu']
    label_all = np.transpose(label_all)
    label_SC2 = np.array(label_all, np.float32) / 255

    edge_all = eg['b_mode_image_quantitative']
    edge_all = np.transpose(edge_all)
    edge_all = np.array(edge_all, np.float32) / 255

    model = UNetwork(args.input_shape, args.label_shape,drop_out=True)
    print("model initialized")
    total_parameters()
    sess = tf.Session()
    summary_placeholders, update_ops, summary_op = setup_summary(["Training Accuracy", "Validation Accuracy"])
    summary_writer = tf.summary.FileWriter('summary/', sess.graph)
    sess.run(tf.global_variables_initializer())
    t_vars = tf.trainable_variables()
    G_vars = [var for var in t_vars if 'encoder' in var.name]

    saver = tf.train.Saver(max_to_keep=None)
    num_images =10000
    batch_size = args.batch_size
    total_steps = int(num_images)+1
    for epoch in range(0,args.epoch):
        if args.drop_out == "True": model.drop_out = True
        loss_sum = 0
        acc_sum = 0
        b = 0
        pr_at, pr_sos, pr_sc1, pr_sc2 = [], [], [], []
        pr_at_valid, pr_sos_valid, pr_sc1_valid, pr_sc2_valid = [], [], [], []
        for i in range(total_steps):
            ind = random.randrange(1, 18000)

            data1 = np.reshape(preprocess_image_RF(RF_all[:, :, 0, ind]),
                               [1, 128, 3018, 1])
            data2 = np.reshape(preprocess_image_RF(RF_all[:, :, 1, ind]),
                               [1, 128, 3018, 1])
            data3 = np.reshape(preprocess_image_RF(RF_all[:, :, 2, ind]),
                               [1, 128, 3018, 1])
            data4 = np.reshape(preprocess_image_RF(RF_all[:, :, 3, ind]),
                               [1, 128, 3018, 1])
            data5 = np.reshape(preprocess_image_RF(RF_all[:, :, 4, ind]),
                               [1, 128, 3018, 1])
            data6 = np.reshape(preprocess_image_RF(RF_all[:, :, 5, ind]),
                               [1, 128, 3018, 1])
            data7 = np.reshape(preprocess_image_RF(RF_all[:, :, 6, ind]),
                               [1, 128, 3018, 1])
            icd = random.randrange(0, 4)
            cond = np.zeros((1, 4))
            if icd == 0:
                label_batch = np.reshape(np.transpose(label_AT[:, :, ind]), [1, 128, 128, 1])
                cond[0, icd] = 1
            elif icd == 1:
                label_batch = np.reshape(np.transpose(label_SOS[:, :, ind]), [1, 128, 128, 1])
                cond[0, icd] = 1
            elif icd == 2:
                label_batch = np.reshape(np.transpose(label_SC1[:, :, ind]), [1, 128, 128, 1])
                cond[0, icd] = 1
            elif icd == 3:
                label_batch = np.reshape(np.transpose(label_SC2[:, :, ind]), [1, 128, 128, 1])
                cond[0, icd] = 1
            edge_batch = np.reshape(np.transpose(edge_all[:, :, random.randrange(0, 6), ind]), [1, 128, 128, 1])

            if random.randrange(1, 100) % 2 ==0:
                data1, data2, data3, data4, data5, data6, data7, label_batch,edge_batch = AUG_RF(data1, data2, data3, data4, data5, data6, data7, label_batch,edge_batch)
                if np.max(edge_batch)>0:
                    edge_batch=edge_batch/np.max(edge_batch)

            data1=np.array(np.round(data1*255),dtype=np.float32)/255
            data2 = np.array(np.round(data2 * 255), dtype=np.float32) / 255
            data3 = np.array(np.round(data3 * 255), dtype=np.float32) / 255
            data4= np.array(np.round(data4 * 255), dtype=np.float32) / 255
            data5 = np.array(np.round(data5 * 255), dtype=np.float32) / 255
            data6 = np.array(np.round(data6 * 255), dtype=np.float32) / 255
            data7 = np.array(np.round(data7 * 255), dtype=np.float32) / 255

            train_sample, _, loss_val = sess.run([model.logits, model.training, model.loss], feed_dict = {model.input1:np.float32(data1),model.input2:np.float32(data2),model.input3:np.float32(data3),model.input4:np.float32(data4), model.input5:np.float32(data5),model.input6:np.float32(data6),model.input7:np.float32(data7),model.labels:np.float32(label_batch),model.bmode:np.float32(edge_batch),model.cond_ph: np.float32(cond)})
            if icd == 0:
                pr_at.append(PSNR(label_batch, train_sample))
            elif icd == 1:
                pr_sos.append(PSNR(label_batch, train_sample))
            elif icd == 2:
                pr_sc1.append(PSNR(label_batch, train_sample))
            elif icd == 3:
                pr_sc2.append(PSNR(label_batch, train_sample))
            loss_sum += loss_val
            b+=batch_size

            if (i) % 10 == 0:
                ind_valid = random.randrange(18000, 19000)
                data1_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 0, ind_valid]),
                                         [1, 128, 3018, 1])
                data2_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 1, ind_valid]),
                                         [1, 128, 3018, 1])
                data3_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 2, ind_valid]),
                                         [1, 128, 3018, 1])
                data4_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 3, ind_valid]),
                                         [1, 128, 3018, 1])
                data5_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 4, ind_valid]),
                                         [1, 128, 3018, 1])
                data6_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 5, ind_valid]),
                                         [1, 128, 3018, 1])
                data7_valid = np.reshape(preprocess_image_RF(RF_all[:, :, 6, ind_valid]),
                                         [1, 128, 3018, 1])

                icd = random.randrange(0, 4)
                cond = np.zeros((1, 4))
                if icd == 0:
                    label_valid = np.reshape(np.transpose(label_AT[:, :, ind_valid]), [1, 128, 128, 1])
                    cond[0, icd] = 1
                elif icd == 1:
                    label_valid = np.reshape(np.transpose(label_SOS[:, :, ind_valid]), [1, 128, 128, 1])
                    cond[0, icd] = 1
                elif icd == 2:
                    label_valid = np.reshape(np.transpose(label_SC1[:, :, ind_valid]), [1, 128, 128, 1])
                    cond[0, icd] = 1
                elif icd == 3:
                    label_valid = np.reshape(np.transpose(label_SC2[:, :, ind_valid]), [1, 128, 128, 1])
                    cond[0, icd] = 1
                edge_batch = np.reshape(np.transpose(edge_all[:, :, random.randrange(0, 6), ind_valid]), [1, 128, 128, 1])
                
                if np.max(edge_batch)>0:
                    edge_batch=edge_batch/np.max(edge_batch)

                val_sample, val_loss = sess.run([model.logits, model.loss], feed_dict={model.input1: np.float32(data1_valid),
                                                                                       model.input2: np.float32(data2_valid),
                                                                                       model.input3: np.float32(data3_valid),
                                                                                       model.input4: np.float32(data4_valid),
                                                                                       model.input5: np.float32(data5_valid),
                                                                                       model.input6: np.float32(data6_valid),
                                                                                       model.input7: np.float32(data7_valid),
                                                                                       model.bmode: np.float32(
                                                                                           edge_batch),
                                                                                       model.labels: np.float32(
                                                                                           label_valid),model.cond_ph: np.float32(cond)})

                if icd == 0:
                    pr_at_valid.append(PSNR(label_valid, val_sample))
                elif icd == 1:
                    pr_sos_valid.append(PSNR(label_valid, val_sample))
                elif icd == 2:
                    pr_sc1_valid.append(PSNR(label_valid, val_sample))
                elif icd == 3:
                    pr_sc2_valid.append(PSNR(label_valid, val_sample))

            if (i) % 100 == 0:
                print(
                    "Epoch: [%2d] [%5d/%5d] AC: PSNR: %.8f, PSNR_Valid: %.8f ---  SOS: PSNR: %.8f, PSNR_Valid: %.8f --- SC1: PSNR: %.8f, PSNR_Valid: %.8f--- SC2: PSNR: %.8f, PSNR_Valid: %.8f" % (
                        epoch, i, num_images, np.mean(pr_at), np.mean(pr_at_valid), np.mean(pr_sos),
                        np.mean(pr_sos_valid)
                        , np.mean(pr_sc1), np.mean(pr_sc1_valid), np.mean(pr_sc2), np.mean(pr_sc2_valid)))
                
        model.drop_out = False

        f = open("./report/training.txt", 'a')
        f.write("%4d\t%.8f\t%.2f\t%.8f\t%.2f\t%.8f\t%.2f\t%.8f\t%.2f\n" % (epoch, np.mean(pr_at), np.mean(pr_at_valid), np.mean(pr_sos),  np.mean(pr_sos_valid)
                        , np.mean(pr_sc1), np.mean(pr_sc1_valid), np.mean(pr_sc2), np.mean(pr_sc2_valid)))
        f.close()
        summary_stats = [acc_sum/total_steps, float((1 - np.sum(abs(np.float32(label_batch) - val_sample)) / (256 * 256 * int(data1.shape[0]))) * 100)]#, step]
        for i in range(len(summary_stats)):
            sess.run(update_ops[i], feed_dict={summary_placeholders[i]: float(summary_stats[i])})
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, epoch + 1)
        if (epoch + 1) % 10 == 0:
            print("Saving model...")
            saver.save(sess, 'save_model' + "model_" + str(epoch + 1) + ".cptk")
        if (epoch+1) % 1 == 0:
            reshaped_label = np.reshape(label_batch, (128, 128))
            reshaped_sample = np.reshape(train_sample, (128, 128))
            scipy.misc.toimage(reshaped_label, cmin=0.0, cmax=1).save(
                "./samples/train_label_%d.png" % (epoch))
            scipy.misc.toimage(reshaped_sample, cmin=0.0, cmax=1).save(
                "./samples/train_sample_%d.png" % (epoch))


            reshaped_label = np.reshape(label_valid, (128, 128))
            reshaped_sample = np.reshape(val_sample, (128, 128))
            scipy.misc.toimage(reshaped_label, cmin=0.0, cmax=1).save(
                "./samples/batch_label_%d.png" % (epoch))
            scipy.misc.toimage(reshaped_sample, cmin=0.0, cmax=1).save(
                "./samples/batch_sample_%d.png" % (epoch))

