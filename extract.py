import base64
import os


def read_line(line):
    image_id, face_data, mid, image_search_rank, image_url = line.split("\t")
    return base64.b64decode(face_data), mid, image_search_rank


def write_image(filename, data):
    with open(filename, "w+b") as f:
        f.write(data)


def unpack(file_name, output_dir):
    i = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            face_data, mid, image_search_rank = read_line(line)

            img_dir = output_dir
            if i % 10 != 7 and i % 10 != 8 and i % 10 != 9:
                img_dir = os.path.join(img_dir, 'train')
            elif i % 10 != 9:
                img_dir = os.path.join(img_dir, 'valid')
            else:
                img_dir = os.path.join(img_dir, 'test')

            img_dir = os.path.join(img_dir, mid)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)

            img_name = image_search_rank + ".jpg"
            write_image(os.path.join(img_dir, img_name), face_data)

            i += 1
            if i % 1000 == 0:
                print(i, "images finished")

        print("all finished")


def main():
    # file_name = "/Users/yantiz/Desktop/ML课程/MLP/project/TrainData_lowshot.tsv"
    # output_dir = "/Users/yantiz/Desktop/ML课程/MLP/project/Extracted_lowshot"

    file_name = "D:/Lab/mlpproject/TrainData_Base.tsv"
    output_dir = "D:/Lab/mlpproject/Extracted_Base"

    unpack(file_name, output_dir)


if __name__ == '__main__':
    main()
