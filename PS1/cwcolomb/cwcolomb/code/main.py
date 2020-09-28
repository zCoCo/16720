"""Main function for running the system."""

from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    ## Q1.1
    # Create a copy of the opts and modify the number of scales (as requested by 1.1.2):
    import argparse
    opts_112 = argparse.Namespace(**vars(opts))
    opts_112.filter_scales = [1,2,3,4,5]
    
    # Run code for 1.1.2:
    img_path = join(opts_112.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts_112, img)
    util.display_filter_responses(opts_112, filter_responses)

    ## Q1.2
    n_cpu = util.get_num_CPU()
    #visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    #img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    #img = Image.open(img_path)
    #img = np.array(img).astype(np.float32)/255
    #dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    #wordmap = visual_words.get_visual_words(opts, img, dictionary)
    #util.visualize_wordmap(wordmap)

    ## Q2.1-2.4
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    # n_cpu = util.get_num_CPU()
    # conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
