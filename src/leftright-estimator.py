#!/usr/bin/python3

import numpy as np
import yarp
import sys
import pickle as pk
import cv2
import distutils.util

from functions.utilities import read_openpose_data, get_features, create_bottle, draw_on_img
from functions.utilities import IMAGE_WIDTH, IMAGE_HEIGHT, NUM_JOINTS, CLASS_DICT

yarp.Network.init()


class LeftRightEstimator(yarp.RFModule):

    def configure(self, rf):
        model_name = rf.find("leftrightestimator_model_name").asString()
        self.clf = pk.load(open('./functions/' + model_name, 'rb'))
        print('SVM model file: %s' % model_name)
        self.CAMERA_POV = bool(distutils.util.strtobool((rf.find("camera_point_of_view").asString())))
        print('Camera point of view: %s' % str(self.CAMERA_POV))

        # input port for rgb image
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/leftrightestimator/image:i')
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1],
                                            self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/leftrightestimator/image:i'))
        
        # input port for depth
        self.in_port_human_depth = yarp.BufferedPortImageFloat()
        self.in_port_human_depth_name = '/leftrightestimator/depth:i'
        self.in_port_human_depth.open(self.in_port_human_depth_name)
        self.in_buf_human_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.in_buf_human_depth = yarp.ImageFloat()
        self.in_buf_human_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_depth.setExternal(self.in_buf_human_depth_array.data, self.in_buf_human_depth_array.shape[1], self.in_buf_human_depth_array.shape[0])
        print('{:s} opened'.format('/leftrightestimator/depth:i'))

        # input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/leftrightestimator/data:i')
        print('{:s} opened'.format('/leftrightestimator/data:i'))

        # output port for rgb image with prediction
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/leftrightestimator/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1],
                                             self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/leftrightestimator/image:o'))

        # propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/leftrightestimator/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                              self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/leftrightestimator/propag:o'))

        # propag input depth
        self.out_port_propag_depth = yarp.Port()
        self.out_port_propag_depth.open('/leftrightestimator/depth:o')
        self.out_buf_propag_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.out_buf_propag_depth = yarp.ImageFloat()
        self.out_buf_propag_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_depth.setExternal(self.out_buf_propag_depth_array.data, self.out_buf_propag_depth_array.shape[1],
                                              self.out_buf_propag_depth_array.shape[0])
        print('{:s} opened'.format('/leftrightestimator/depth:o'))

        # output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/leftrightestimator/pred:o')
        print('{:s} opened'.format('/leftrightestimator/pred:o'))

        # command port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/leftrightestimator/command:i')
        print('{:s} opened'.format('/leftrightestimator/command:i'))
        self.attach(self.cmd_port)

        self.buffer = ('', (0, 0), '', 0) # centroid, prediction and level of confidence
        self.svm_buffer_size = 3
        self.svm_buffer = []
        self.id_image = '%08d' % 0

        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            print('received command QUIT')
            self.cleanup()
            reply.addString('QUIT command sent')
        elif command.get(0).asString() == 'get':
            print('received command GET')
            reply.copy(create_bottle(self.buffer))
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))

        return True

    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.in_port_human_depth.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        return True

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.in_port_human_depth.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):

        received_image = self.in_port_human_image.read()
        received_depth = self.in_port_human_depth.read(False)

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)
            self.human_image = np.copy(human_image)
            self.id_image = '%08d' % ((int(self.id_image) + 1) % 100000)

            if received_depth:
                self.in_buf_human_depth.copy(received_depth)
                human_depth = np.copy(self.in_buf_human_depth_array)

            received_data = self.in_port_human_data.read()
            if received_data:
                poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                # get features of all people in the image
                data = get_features(poses, conf_poses, faces, conf_faces)
                if data:
                    # predict model
                    # start from 2 because there is the centroid valued in the position [0,1]
                    ld = np.array(data)
                    x = ld[:, 2:(NUM_JOINTS * 2) + 2]
                    c = ld[:, (NUM_JOINTS * 2) + 2:ld.shape[1]]
                    # weight the coordinates for its confidence value
                    wx = np.concatenate((np.multiply(x[:, ::2], c), np.multiply(x[:, 1::2], c)), axis=1)

                    # return a prob value between 0,1 for each class
                    y_classes = self.clf.predict_proba(wx)
                    # take only the person with id 0, we suppose that there is only one person in the scene
                    itP = 0
                    prob = max(y_classes[itP])
                    y_pred = (np.where(y_classes[itP] == prob))[0]

                    if len(self.svm_buffer) == self.svm_buffer_size:
                        self.svm_buffer.pop(0)

                    self.svm_buffer.append([y_pred[0], prob])

                    count_class_0 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(0)
                    count_class_1 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(1)
                    if count_class_1 == count_class_0:
                        y_winner = y_pred[0]
                        prob_mean = prob
                    else:
                        y_winner = np.argmax([count_class_0, count_class_1])
                        prob_values = np.array(
                            [self.svm_buffer[i][1] for i in range(0, len(self.svm_buffer)) if
                             self.svm_buffer[i][0] == y_winner])
                        prob_mean = np.mean(prob_values)

                    # reverse the result depending of camera pov
                    if not self.CAMERA_POV:
                        if y_winner == 0:
                            y_winner = 1
                        elif y_winner == 1:
                            y_winner = 0

                    label = list(CLASS_DICT.keys())[y_winner]
                    pred = create_bottle((self.id_image, (int(ld[itP, 0]), int(ld[itP, 1])), label, prob_mean, received_data.get(0)))
                    human_image = draw_on_img(human_image, self.id_image, (ld[itP, 0], ld[itP, 1]), label, prob_mean)

                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_prediction.write(pred)

                    self.buffer = (self.id_image, (int(ld[itP, 0]), int(ld[itP, 1])), y_winner, prob_mean)
                else:
                    pred = create_bottle((self.id_image, (), 'unknown', -1, ()))
                    human_image = cv2.putText(human_image, 'id: ' + str(self.id_image), tuple([25, 30]), cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6, (0, 0, 255), 2, cv2.LINE_AA)

                    # send in output only the image with prediction set to -1 (invalid value)
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_prediction.write(pred)

                    self.buffer = (self.id_image, (), 'unknown', -1)

            # propag received image
            self.out_buf_propag_image_array[:, :] = self.human_image
            self.out_port_propag_image.write(self.out_buf_propag_image)
            if received_depth:
                # propag received depth
                self.out_buf_propag_depth_array[:, :] = human_depth
                self.out_port_propag_depth.write(self.out_buf_propag_depth)

        return True


if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("LeftRightEstimator")
    rf.setDefaultConfigFile('../app/config/leftrightestimator_conf.ini')
    rf.configure(sys.argv)

    # Run module
    manager = LeftRightEstimator()
    manager.runModule(rf)
