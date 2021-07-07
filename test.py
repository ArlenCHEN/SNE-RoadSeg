import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.util import confusion_matrix, getScores, save_images, compute_scores
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    percent_list = [1, 10, 100]
    for label_percent in percent_list:
        opt.epoch = str(label_percent)

        save_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt, dataset.dataset)
        model.setup(opt)
        model.eval()

        test_loss_iter = []
        epoch_iter = 0
        # conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float)
        running_acc = []
        running_precision = []
        running_recall = []
        running_f_score = []
        running_iou = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataset)):
                model.set_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += opt.batch_size
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()
                save_images(save_dir, model.get_current_visuals(), model.get_image_names(), model.get_image_oriSize(), opt.prob_map)
                gt = np.squeeze(gt)
                pred = np.squeeze(pred)

                rgb = data['rgb_image']
                rgb_img = rgb.detach().cpu().numpy()
                rgb_img = np.squeeze(rgb_img)
                vis_rgb = np.transpose(rgb_img, (1,2,0))

                is_plot = True
                img_color = 'Blues'
                if is_plot:
                    nrow = 1
                    ncol = 3
                    show_color = 'Blues'
                    fig = plt.figure(figsize=(30, 7.6))
                    gs = gridspec.GridSpec(nrow, ncol,
                                            wspace=0.0, hspace=0.0, 
                                            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                                            left=0.5/(ncol+1), right=1-0.5/(ncol+1))

                    ax1 = plt.subplot(gs[0,0])
                    ax1.imshow(vis_rgb, cmap=img_color)
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])
                    ax1.axis('off')

                    ax1 = plt.subplot(gs[0,1])
                    ax1.imshow(gt, cmap=img_color)
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])
                    ax1.axis('off')

                    ax1 = plt.subplot(gs[0,2])
                    ax1.imshow(pred, cmap=img_color)
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])
                    ax1.axis('off')
                    plt.savefig('./test_imgs/'+opt.epoch+'-'+str(i)+'.png')
                    # plt.show()

                acc, precision, recall, f_score, iou = compute_scores(pred, gt)
                running_acc += [acc]
                running_precision += [precision]
                running_recall += [recall]
                running_f_score += [f_score]
                running_iou += [iou]
            acc_mean = np.mean(running_acc)
            acc_std = np.std(running_acc)
            precision_mean = np.mean(running_precision)
            precision_std = np.std(running_precision)
            recall_mean = np.mean(running_recall)
            recall_std = np.std(running_recall)
            f_score_mean = np.mean(running_f_score)
            f_score_std = np.std(running_f_score)
            iou_mean = np.mean(running_iou)
            iou_std = np.std(running_iou)

            #     # Resize images to the original size for evaluation
            #     image_size = model.get_image_oriSize()
            #     oriSize = (image_size[0].item(), image_size[1].item())
            #     gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            #     pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
                
            #     conf_mat += confusion_matrix(gt, pred, dataset.dataset.num_labels)

            #     test_loss_iter.append(model.loss_segmentation)
            #     print('Epoch {0:}, iters: {1:}/{2:}, loss: {3:.3f} '.format(opt.epoch,
            #                                                                 epoch_iter,
            #                                                                 len(dataset) * opt.batch_size,
            #                                                                 test_loss_iter[-1]), end='\r')

            # avg_test_loss = torch.mean(torch.stack(test_loss_iter))
            # print ('Epoch {0:} test loss: {1:.3f} '.format(opt.epoch, avg_test_loss))
            # globalacc, pre, recall, F_score, iou = getScores(conf_mat)
            print ('Label precent {0:} glob acc mean : {1:.3f}, pre mean: {2:.3f}, recall mean: {3:.3f}, F_score mean: {4:.3f}, IoU mean: {5:.3f}'.format(opt.epoch, acc_mean, precision_mean, recall_mean, f_score_mean, iou_mean))
            print ('Label precent {0:} glob acc std : {1:.3f}, pre std: {2:.3f}, recall std: {3:.3f}, F_score std: {4:.3f}, IoU std: {5:.3f}'.format(opt.epoch, acc_std, precision_std, recall_std, f_score_std, iou_std))
