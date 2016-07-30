import numpy as np
from ImageLoader import ImageLoader
from PairedDual import PairedDual
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from Evaluator import Evaluator
import time
import os
from xlwt import Workbook
import xlwt



def main():
    learners = []
    d_unary = 65
    num_states = 2
    d_edge = 11
    max_height = 10
    max_width = 10
    num_training_images = 2
    num_testing_images = 2
    max_iters = [5, 10, 20]
    inc = 'true'
    plot = 'false'
    objective_types = ['primal', 'dual']
    inference_types = {'BP': MatrixBeliefPropagator, 'TRBP': MatrixTRBeliefPropagator}
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))

    loader = ImageLoader(max_height, max_width)

    images, models, labels, names = loader.load_all_images_and_labels(path+'/test/train', 2, num_training_images)

    # chart frame:
    wb = Workbook()
    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_RIGHT
    style = xlwt.XFStyle()
    style.alignment = alignment
    style.num_format_str = '#,##0.0000'

    # style0 = xlwt.easyxf(num_format_str='#,##0.0000')

    sheet1 = wb.add_sheet('Results')

    sheet1.write(0,0,'Max_iter', style)
    sheet1.write(0,1,'TRBP or BP', style)
    sheet1.write(0,2,'Primal or Dual', style)
    sheet1.write(0,3,'training error', style)
    sheet1.write(0,4,'training incon', style)
    sheet1.write(0,5,'testing error', style)
    sheet1.write(0,6,'testing incon', style)
    sheet1.write(0,7,'training time', style)

    sheet1.col(0).width = 3000
    sheet1.col(1).width = 3000
    sheet1.col(2).width = 3000
    sheet1.col(3).width = 3000
    sheet1.col(4).width = 3000
    sheet1.col(5).width = 3000
    sheet1.col(6).width = 3000
    sheet1.col(7).width = 3000

    n = 1
    for max_iter in max_iters:
        for inference_type_name in inference_types:
            inference_type = inference_types[inference_type_name]
            for objective_type in objective_types:
                start = time.time()
                sheet1.write(n,0,max_iter)
                sheet1.write(n,1,inference_type_name, style)
                sheet1.write(n,2,objective_type, style)

                if objective_type is 'dual':
                    learner = PairedDual(inference_type, max_iter)
                else:
                    learner = Learner(inference_type)
                learners.append(learner)
                learner.set_regularization(0.0, 1.0)


                for model, states in zip(models, labels):
                    learner.add_data(states, model)

                for bp in learner.belief_propagators_q:
                 bp.set_max_iter(max_iter)
                for bp in learner.belief_propagators:
                 bp.set_max_iter(max_iter)

                weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

                new_weights = learner.learn(weights)

                unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
                pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states ** 2))
                # print("Unary weights:\n" + repr(unary_mat))
                # print("Pairwise weights:\n" + repr(pair_mat))

                elapsed = time.time() - start
                print(
                "Time to train the weights: %f. configuration: max_iter: %d, inference type: %s, objective type: %s" % (
                elapsed, max_iter, inference_type_name, objective_type))

            # Evaluations

                Eval = Evaluator(max_height, max_width)
                if num_training_images > 0:
                    print("Training:")
                    if inc == "true":
                        train_errors, train_total_inconsistency = Eval.evaluate_training_images(images, models, labels,
                                                                                                names, new_weights, 2,
                                                                                                num_training_images,
                                                                                                inference_type,
                                                                                                max_iter, inc, plot)
                    else:
                        train_errors = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2,
                                                                     num_training_images,
                                                                     inference_type, max_iter, inc, plot)
                    print ("Average Train Error rate: %f" % train_errors)
                    print "\n"

                sheet1.write(n,3,train_errors, style)
                if inc == "true":
                    sheet1.write(n,4,train_total_inconsistency, style)
                else:
                    sheet1.write(n, 4, 'Not calculated', style)


                if num_testing_images > 0:
                    print("Test:")
                    if inc == "true":
                        test_errors, test_total_inconsistency = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images, inference_type, max_iter, inc, plot)
                    else:
                        test_errors = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images, inference_type, max_iter, inc, plot)
                    print ("Average Test Error rate: %f" % test_errors)

                    sheet1.write(n,5, test_errors, style)
                    if inc == "true":
                        sheet1.write(n,6, test_total_inconsistency, style)
                    else:
                        sheet1.write(n, 6, 'Not calculated', style)

                else:
                    sheet1.write(n, 5, 'Not calculated', style)
                    sheet1.write(n, 6, 'Not calculated', style)

                sheet1.write(n, 7, elapsed, style)


                n += 1

    wb.save('Results.xls')


if __name__ == "__main__":
    main()
