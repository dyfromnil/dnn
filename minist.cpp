#include "network.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <time.h>
using namespace std;

uint32_t transform(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_image(vector<Input> &data_set, double percent, string path_image, string path_label)
{
    if (percent <= 0 || percent > 1)
    {
        cerr << "percent error!" << endl;
        exit(0);
    }

    uint32_t magic;     //文件中的魔术数(magic number)
    uint32_t num_image; //mnist图像集文件中的图像数目
    uint32_t num_label; //mnist标签集文件中的标签数目
    uint32_t rows;      //图像的行数
    uint32_t cols;      //图像的列数

    ifstream image(path_image, ios::in | ios::binary);
    ifstream label(path_label, ios::in | ios::binary);

    if (!image.is_open())
        cout << "read_image_error" << endl;
    if (!label.is_open())
        cout << "read_label_error" << endl;

    image.read(reinterpret_cast<char *>(&magic), 4);
    if (transform(magic) != 2051)
        cout << "image !2051" << endl;
    label.read(reinterpret_cast<char *>(&magic), 4);
    if (transform(magic) != 2049)
        cout << "label !2049" << endl;

    image.read(reinterpret_cast<char *>(&num_image), 4);
    label.read(reinterpret_cast<char *>(&num_label), 4);
    if (transform(num_image) != transform(num_label))
        cout << "number of image&label not match!" << endl;

    image.read(reinterpret_cast<char *>(&rows), 4);
    rows = transform(rows);
    image.read(reinterpret_cast<char *>(&cols), 4);
    cols = transform(cols);

    int num_data = int(percent * transform(num_image));

    // read image
    for (int i = 0; i < num_data; ++i)
    {
        Input data_;
        unsigned char vec_ima[rows * cols];
        unsigned char num;
        image.read(reinterpret_cast<char *>(&vec_ima), rows * cols);
        label.read(reinterpret_cast<char *>(&num), 1);

        vector<double> vec_image(vec_ima, vec_ima + rows * cols);
        for (double &value : vec_image)
            value /= 255;
        vector<double> vec_label(10, 0);
        vec_label[num] = 1;
        data_set.push_back({vec_image, vec_label});
    }
}

void diff(struct timespec *start, struct timespec *end, struct timespec *interv)
{
    if ((end->tv_nsec - start->tv_nsec) < 0)
    {
        interv->tv_sec = end->tv_sec - start->tv_sec - 1;
        interv->tv_nsec = 1e9 + end->tv_nsec - start->tv_nsec;
    }
    else
    {
        interv->tv_sec = end->tv_sec - start->tv_sec;
        interv->tv_nsec = end->tv_nsec - start->tv_nsec;
    }
}

void timediff(long left_s, long &time_h, long &time_m, long &time_s)
{
    time_s = left_s % 60;
    time_m = (left_s / 60) % 60;
    time_h = (left_s / 3600) % 24;
}

void time_format(char *buffer, int h, int m, int s)
{
    time_t rawtime;
    struct tm *timeinfo;

    timeinfo = localtime(&rawtime);
    timeinfo->tm_hour = h;
    timeinfo->tm_min = m;
    timeinfo->tm_sec = s;

    strftime(buffer, 20, "%Hh %Mmin %Ss", timeinfo);
}

int main(int agrc, char **argv)
{
    vector<Input> train_dataset;
    vector<Input> test_dataset;

    read_image(train_dataset, 1, "./data_set/train-images.idx3-ubyte", "./data_set/train-labels.idx1-ubyte");
    read_image(test_dataset, 1, "./data_set/t10k-images.idx3-ubyte", "./data_set/t10k-labels.idx1-ubyte");
    int num_of_trainset = train_dataset.size();
    int num_of_testset = test_dataset.size();

    double lr = 0.7;
    Network network(lr);

    // network.append_layer(784, 128, "sigmoid");
    // network.append_layer(128, 10, "softmax");
    network.append_layer(784, 10, "softmax");

    // training....
    network.weight_init();

    int epoch = 20;
    int decay_start_epoch = epoch * 0.7;
    int batch_size = 128;
    int batch_total = int(train_dataset.size()) / batch_size;
    int batch_left = epoch * batch_total;

    timespec prev_time;
    timespec now;
    clock_gettime(0, &prev_time);

    for (int epo = 0; epo < epoch; ++epo)
    {
        for (int batch = 0; batch < batch_total; ++batch)
        {
            double loss_average = 0;
            network.zero_grad();
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; ++i)
            {
                double loss = 0;
                network.forward(train_dataset[i]);
                network.backward(train_dataset[i], loss);
                loss_average += loss;
            }
            loss_average /= train_dataset.size();

            clock_gettime(0, &now);
            long time_h, time_m, time_s;
            timespec interv;
            diff(&prev_time, &now, &interv);
            long left_s = batch_left * (interv.tv_sec * 1e9 + interv.tv_nsec) / 1e9;
            timediff(left_s, time_h, time_m, time_s);
            char buffer[20];
            time_format(buffer, time_h, time_m, time_s);
            prev_time = now;

            cout.unsetf(ios::fixed);
            cout << "[train:" << num_of_trainset << "][test: " << num_of_testset << "][batch size:" << batch_size << "][lr: " << network.get_lr() << "][epoch:" << epo << "/" << epoch << "][batch: " << batch << "/" << train_dataset.size() / batch_size << fixed << setprecision(8) << "][loss:" << loss_average << "]"
                 << "[left time: " << buffer << "]" << endl;

            network.step();
            --batch_left;
        }
        double decay_rate = 1 - double(max(0, epo - decay_start_epoch)) / (double)(epoch - decay_start_epoch);
        network.set_lr(decay_rate * lr);
    }

    network.predict_minist(test_dataset);

    return 0;
}