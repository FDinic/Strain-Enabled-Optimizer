#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import argparse
import sys
import time
from pymatgen.core.structure import Structure
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from cgcnn.data import OPTData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet
import gc


def main(argv):
    start_time = time.time()  # start timing

    parser = argparse.ArgumentParser(description="Crystal gated neural networks")
    parser.add_argument("modelpath", help="path to the trained model.")
    parser.add_argument("source", help="path to source cif file")
    parser.add_argument("-c", "--cifpath", default="data/Opt/", metavar='N',
                        type=str, help="path to the directory of CIF files.")
    parser.add_argument("-d", "--disable-cuda", action='store_true', help='Disable CUDA')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=250, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-i', "--input-path", default="in/bulk.cif", metavar='N',
                        type=str, help="input cif location")
    parser.add_argument("-l", "--limits", default=[-0.015, 0.015], metavar='L', nargs=2,
                        type=float, help="distortion limits")
    parser.add_argument("-n", "--n_coords", default=1, type=int,
                        help="number of coordinates to modify per cycle")
    parser.add_argument("-f", "--fractional", action="store_true", help="limits are fractional")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument("-o", "--output", default="test_results.csv", type=str,
                        help="location of test_results.csv")
    parser.add_argument("-t", "--threshold", default=0.0001, type=float,
                        metavar="N", help="min difference threshold")
    parser.add_argument("-m", "--max-step", default=50, type=int, help="maximum number of iterations below threshold")
    parser.add_argument("-a", "--avg-step", action="store_true", help="average vector steps when combining")
    parser.add_argument("-r", "--random-generation", action="store_true", help="stepwise generation, each atom is "
                                                                                "perturbed sequentially from the top "
                                                                                "of list")
    parser.add_argument("-z", "--cell_opt", action="store_true", help="Performed lattice celloptimization")
    parser.add_argument("-y", "--cell_opt_frequency", default=10000, type=int, metavar='N', help="Performed lattice celloptimization")

    args = parser.parse_args(argv[1:])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    normalizer = Normalizer(torch.zeros(3))

    print("ML optimizer using CGCNN")
    print("batch size: {}, step boundary {}, {}, random generation: {}".format(args.batch_size, args.limits[0],
                                                                               args.limits[1], args.random_generation))

    try:
        print("=> loading model params '{}'".format(args.modelpath))
        model_checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        normalizer.load_state_dict(model_checkpoint['normalizer'])
    except FileNotFoundError:
        print("=> no model params found at '{}'".format(args.modelpath))
        exit(2)
    print("Model load time: %s" % (time.time() - start_time))
    start_time = time.time()

    src_struct = Structure.from_file(args.source)
    print("Src load time: %s" % (time.time() - start_time))
    start_time = time.time()

    dataset = OPTData(args.cifpath, src_struct, args.batch_size, args.limits,
                      n_coords=args.n_coords, mod_cart=args.fractional, clone=False)
    print("Cif load time: %s" % (time.time() - start_time))
    start_time = time.time()

    structures, _, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task == 'classification' else False)
    model.load_state_dict(model_checkpoint['state_dict'])

    if args.cuda:
        model.cuda()
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    model.eval()
    print("Model creation time: %s" % (time.time() - start_time))
    start_time = time.time()
    if args.random_generation:
        gen_length = args.batch_size
    else:
        gen_length = len(structures[0]) * 6

    # generate initial energy
    dataset = OPTData(args.cifpath, src_struct, clone=False)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_pool, pin_memory=args.cuda)
    _, init_preds = validate(args, val_loader, model, model_args, criterion, normalizer, args.output)
    prev_energ = init_preds[0]
    # Sentinels
    counter = 0
    step = 0
    print("Initial prediction time: %s" % (time.time() - start_time))
    loopstart = time.time()
    gcount = 0
    ccount=0
    limits = np.array(args.limits)
    # Output file handling
    with open("output.out", "w") as rtally:
        rtally.write("Step, Step min, E step, dE, Min current, N lower [{}], Gradient\n".format(gen_length))
        rtally.flush()
        with open("gradient.out", "w") as gtally:
            gtally.write("Gradient List\n")
            gtally.flush()
            # Optloop
            while 1:
                start_time = time.time()
                # generate dataset
                dataset = OPTData(args.cifpath, src_struct, gen_length, limits,
                                  n_coords=args.n_coords, mod_cart=(not args.fractional), rand=args.random_generation)
                print("Dataset creation time: %s" % (time.time() - start_time))
                start_time = time.time()
                # put into CGCNN model
                test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate_pool,
                                         pin_memory=args.cuda)
                ids, preds = validate(args, test_loader, model, model_args, criterion, normalizer, args.output)
                print("Prediction time: %s" % (time.time() - start_time))
                start_time = time.time()
                # find structures with lower energy than previous step
                out_dat = pd.DataFrame({"ids": ids, "preds": preds})
                low_struct = out_dat[out_dat.preds < prev_energ]
                min_id = out_dat.preds.idxmin()
                min_e = out_dat.preds[min_id]
                print(min_id)
                if not low_struct.empty:  # if found
                    avg_struct = src_struct.copy()  # prepare structure for next step
                    gradient = pd.DataFrame()  # coord shifts
                    for i, row in low_struct.iterrows():  # collate up individual shifts
                        shifts = dataset.struct(row.ids)[3]
                        for j, k in shifts:
                            gradient = gradient.append({"i": j, "coord": k}, ignore_index=True)
                    if args.avg_step:  # sum or average shifts
                        gradient = gradient.groupby("i")["coord"].apply(np.mean)
                    else:
                        gradient = gradient.groupby("i")["coord"].apply(np.sum)
                    gcount += 1
                    gtally.write("{}: x, y, z\n".format(gcount))
                    for i in gradient.index:
                        gtally.write(str(int(i))+","+",".join(["{:17.12f}".format(j) for j in gradient[i]])+"\n")
                    while 1: # descent loop
                        # apply gradient until energy is higher
                        for i in gradient.index:  # apply shifts to structure
                            avg_struct.translate_sites([int(i)], gradient[i], args.fractional)
                        # find energy of step
                        avgdat = OPTData(args.cifpath, avg_struct, clone=False)
                        val_loader = DataLoader(avgdat, batch_size=1, shuffle=False, num_workers=args.workers,
                                                collate_fn=collate_pool, pin_memory=args.cuda)
                        _, avg_preds = validate(args, val_loader, model, model_args, criterion, normalizer, args.output)
                        avg_energ = avg_preds[0]
                        if prev_energ < avg_energ:  # exit if energy is higher
                            break
                        # outputs
                        avg_struct.to(fmt="cif", filename="chk.cif")
                        diff = prev_energ - avg_energ
                        print("Energy analysis time: %s" % (time.time() - start_time))
                        start_time = time.time()
                        print("{}: Min E: {:4} Step E: {:4.12f}, dE: {:4.12f}, best: {:4.12f}".format(step, min_e,
                                                                                                      avg_energ, diff,
                                                                                                      prev_energ))
                        rtally.write("{}, {:4}, {:4.12f}, {:4.12f}, {:4.12f}, {}, {}\n".format(step, min_e, avg_energ, diff,
                                                                                               prev_energ, len(low_struct),
                                                                                               gcount))
                        step += 1
                        rtally.flush()
                        # update energy
                        prev_energ = avg_energ
                        src_struct = avg_struct
                    # if gradient is success add here
                else:
                    gcount += 1
                    avg_energ = min_e
                    # outputs
                    diff = prev_energ - avg_energ
                    print("Energy analysis time: %s" % (time.time() - start_time))
                    start_time = time.time()
                    print("{}: Min E: {:4} Step E: {:4.12f}, dE: {:4.12f}, best: {:4.12f}".format(step, min_e, avg_energ,
                                                                                                  diff,
                                                                                                  prev_energ))
                    rtally.write(
                        "{}, {:4}, {:4.12f}, {:4.12f}, {:4.12f}, {}, {}\n".format(step, min_e, avg_energ, diff, prev_energ,
                                                                                  len(low_struct), gcount))
                    step += 1
                    rtally.flush()
                    # update energy
                    if diff > 0:
                        prev_energ = avg_energ
                    # if gradient is fail add here
                
				# Cell Opt every Nth step
				#HI FIX THE DASHESFOR CELL OPT
                if args.cell_opt:
                    if step % args.cell_opt_frequency == 0:
						#gen_lenght, replace with arg.batch
                        dataset = OPTData(args.cifpath, src_struct, gen_length, limits, n_coords=args.n_coords, mod_cart=(not args.fractional), rand=args.random_generation, cell_opt=True)
						# put into CGCNN model
                        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
												 num_workers=args.workers, collate_fn=collate_pool,
												 pin_memory=args.cuda)
                        ids, preds = validate(args, test_loader, model, model_args, criterion, normalizer, args.output)

						# find structures with lower energy than previous step
                        out_dat = pd.DataFrame({"ids": ids, "preds": preds})
                        cell_opt_low_struct = out_dat[out_dat.preds < prev_energ]
                        min_id = out_dat.preds.idxmin()
                        min_e = out_dat.preds[min_id]
                        print(min_id)
						
                        if not cell_opt_low_struct.empty:  # if found then make it into new structure.
                            structure_fea, target, cif_id, crystal = dataset[min_id]
                            src_struct = crystal
							#take lowest structure, make into new one
							
                            avg_energ = min_e
							#print details in output.
                            ccount += 1							
							
                            avg_struct = src_struct  # prepare structure for next step

                            avg_struct.to(fmt="cif", filename="chk.cif")
								
                            diff = prev_energ - avg_energ
                            print("Energy analysis time: %s" % (time.time() - start_time))
                            start_time = time.time()
                            print("we did a cell opt step!!!!!, yay!!!")
                            print("{}: Min E: {:4} Step E: {:4.12f}, dE: {:4.12f}, best: {:4.12f}".format(step, min_e,avg_energ, diff,prev_energ))
                            rtally.write("{}, {:4}, {:4.12f}, {:4.12f}, {:4.12f}, {}, ,,{}, is cellopt \n".format(step, min_e, avg_energ, diff,prev_energ, len(cell_opt_low_struct),ccount))
                            step += 1
                            rtally.flush()
								# update energy
                            prev_energ = avg_energ
                            src_struct = avg_struct
							# if gradient is success add here
                        else:
                            ccount += 1
                            avg_energ = min_e
							# outputs
                            diff = prev_energ - avg_energ
                            print("Energy analysis time: %s" % (time.time() - start_time))
                            start_time = time.time()
                            print ("we tried a cell opt, but it didnt work :(")
                            print("{}: Min E: {:4} Step E: {:4.12f}, dE: {:4.12f}, best: {:4.12f}".format(step, min_e, avg_energ,
																										  diff,
																										  prev_energ))
                            rtally.write(
								"{}, {:4}, {:4.12f}, {:4.12f}, {:4.12f}, {}, ,{},tried cellopt\n".format(step, min_e, avg_energ, diff, prev_energ,
																						  len(cell_opt_low_struct), ccount))
                            step += 1
                            rtally.flush()
							# update energy
                            if diff > 0:
                                prev_energ = avg_energ
					 
				# cleanup for next iter
                del dataset
                del test_loader
                gc.collect()
                print("Output time: %s" % (time.time() - start_time))
                # if -args.threshold < diff < args.threshold:
                #     counter += 1
                #     if counter > args.max_step:
                #         break
                if len(low_struct) <= 0:
                    counter += 1
                    if counter > args.max_step:
                        break
                else:
                    counter = 0
    print(" generation time: %s" % (time.time() - loopstart))
    gc.collect()
    torch.cuda.empty_cache()
    src_struct.to(fmt="cif", filename="output.cif")


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def validate(args, val_loader, model, model_args, criterion, normalizer, test_results, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    test_targets = []
    test_preds = []
    test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if model_args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            test_pred = torch.exp(output.data.cpu())
            test_target = target
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
            test_targets += test_target.view(-1).tolist()
        test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))
        gc.collect()

    if test:
        star_label = '**'
        import csv
        with open(test_results, 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    # if model_args.task == 'regression':
    #     print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
    #                                                     mae_errors=mae_errors))
    #     return mae_errors.avg
    # else:
    #     print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
    #                                              auc=auc_scores))
    #     return auc_scores.avg
    return test_cif_ids, test_preds


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


if __name__ == '__main__':
    main(sys.argv)
