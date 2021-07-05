import argparse
import todo
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", default='train', type = str, help="train, test, img_aug")
    parser.add_argument("--input_shape", default=[128, 3018, 1], type=list)
    parser.add_argument("--label_shape", default=[128, 128, 1], type=list)
    parser.add_argument("--batch_size", default=8, type = int)
    parser.add_argument("--epoch", default=10000, type = int)
    parser.add_argument("--model_num", default="1000", type = str)
    parser.add_argument("--drop_out", default="False", type = str)

    parser.add_argument("--save_model_rate", default=500, type = int)
    parser.add_argument("--aug_size", default=30, type = int)
    ##############################################
    args = parser.parse_args()
    ##############################################
    if args.todo == "train": todo.train(args)
    if args.todo == "test": todo.test(args)

if __name__ == "__main__":
    main()
