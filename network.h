#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
using namespace std;

typedef struct _input_sample
{
    vector<double> data;
    vector<double> label;
} Input;

typedef struct _node
{
    double in_value, ac_value, bias, b_gradient, delta;
    vector<double> weight, w_gradient;
} Node;

class Layer
{
    typedef void (Layer::*pf)();
    typedef double (*pfb)(double);

public:
    Layer(int in_num, int out_num, string ac_fun);
    void _layer_weight_init(double n);
    void _layer_zero_grad();

    inline void sigmoid()
    {
        for (Node &node : nodes)
            node.ac_value = 1 / (1 + exp(-node.in_value));
    }

    static inline double sigmoid_derivative(double z)
    {
        return exp(-z) / pow(1 + exp(-z), 2);
    }

    inline void relu()
    {
        for (Node &node : nodes)
            node.ac_value = node.in_value >= 0 ? node.in_value : 0;
    }

    static inline double relu_derivative(double z)
    {
        return z >= 0 ? 1 : 0;
    }

    inline void leak_relu()
    {
        for (Node &node : nodes)
            node.ac_value = node.in_value >= 0 ? node.in_value : 0.2 * node.in_value;
    }

    static inline double leak_relu_derivative(double z)
    {
        return z >= 0 ? 1 : 0.2;
    }

    inline void softmax()
    {
        double sum = 0;
        for (Node &node : nodes)
            sum += exp(node.in_value);
        for (Node &node : nodes)
            node.ac_value = exp(node.in_value) / sum;
    }

    inline void activate_function()
    {
        (this->*ac_fun)();
        // (*this.*(*this).ac_fun)();
    }

    // inline double activate_function_backward(double z)
    // {
    //     return (this->*ac_fun_backward)(z);
    // }

private:
    inline double get_random();

public:
    int input_num, nodes_num;
    vector<Node> nodes;
    pf ac_fun = nullptr;
    pfb ac_fun_backward = nullptr;
    string ac_fu;
};

Layer::Layer(int in_num, int out_num, string acfun) : input_num(in_num), nodes_num(out_num), ac_fu(acfun)
{
    Node node;
    for (int i = 0; i < in_num; ++i)
    {
        node.weight.emplace_back(0.f);
        node.w_gradient.emplace_back(0.f);
    }
    nodes.insert(nodes.begin(), out_num, node);
    if (acfun == "sigmoid")
    {
        ac_fun = &Layer::sigmoid;
        ac_fun_backward = &Layer::sigmoid_derivative;
    }
    else if (acfun == "relu")
    {
        ac_fun = &Layer::relu;
        ac_fun_backward = &Layer::relu_derivative;
    }
    else if (acfun == "leak_relu")
    {
        ac_fun = &Layer::leak_relu;
        ac_fun_backward = &Layer::leak_relu_derivative;
    }
    else
    {
        ac_fun = &Layer::softmax;
    }
}

void Layer::_layer_weight_init(double n)
{
    if (ac_fu == "relu" || ac_fu == "leak_relu")
    {
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            std::random_device rd;
            std::default_random_engine rng{rd()};
            normal_distribution<> norm{0, sqrt(2 / n)};
            nodes[i].weight.assign(input_num, norm(rd));
            nodes[i].bias = 0.f;
        }
    }
    else
    {
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            nodes[i].weight.assign(input_num, get_random());
            nodes[i].bias = 0.f;
        }
    }
}

void Layer::_layer_zero_grad()
{
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        nodes[i].w_gradient.assign(input_num, 0.f);
        nodes[i].b_gradient = 0.f;
    }
}

double Layer::get_random()
{
    return ((2.0 * (double)rand() / RAND_MAX) - 1);
}

class Network
{
    typedef double (*p)(double);

public:
    Network(double learning_rate);
    void append_layer(int in_num, int out_num, string ac_fun = "sigmoid");
    void forward(Input &input);
    void backward(Input &input, double &loss);
    void predict(vector<Input> &test_group);
    void predict_minist(vector<Input> &test_group);
    void step();
    void weight_init();
    void zero_grad();
    void set_lr(double learning_rate);
    inline double get_lr() { return lr; }

public:
    vector<Layer> layers;
    int step_len = 0;
    double lr = 0.7;
};

Network::Network(double learning_rate) { lr = learning_rate; }

void Network::append_layer(int in_num, int out_num, string ac_fun)
{
    layers.emplace_back(in_num, out_num, ac_fun);
}

void Network::forward(Input &input)
{
    // first hidden layer
    for (int i = 0; i < layers[0].nodes_num; ++i)
    {
        layers[0].nodes[i].in_value = 0.f;
        for (int j = 0; j < layers[0].input_num; ++j)
        {
            layers[0].nodes[i].in_value += layers[0].nodes[i].weight[j] * input.data[j];
        }
        layers[0].nodes[i].in_value += layers[0].nodes[i].bias;
    }
    // (layers[0]->*ac_fun)();
    layers[0].activate_function();

    // remained layers
    for (size_t k = 1; k < layers.size(); ++k)
    {
        for (int i = 0; i < layers[k].nodes_num; ++i)
        {
            layers[k].nodes[i].in_value = 0.f;
            for (int j = 0; j < layers[k].input_num; ++j)
            {
                layers[k].nodes[i].in_value += layers[k].nodes[i].weight[j] * layers[k - 1].nodes[j].ac_value;
            }
            layers[k].nodes[i].in_value += layers[k].nodes[i].bias;
        }
        // layers[k].ac_fun();
        layers[k].activate_function();
    }
}

void Network::backward(Input &input, double &loss)
{
    ++step_len;

    /* 
        compute delta
    */

    // output layer
    for (int i = 0; i < layers[layers.size() - 1].nodes_num; ++i)
    {
        // loss += fabs(input.label[i] * log(layers[layers.size() - 1].nodes[i].ac_value) + (1 - input.label[i]) * (1 - log(layers[layers.size() - 1].nodes[i].ac_value)));
        loss += fabs(input.label[i] - layers[layers.size() - 1].nodes[i].ac_value);
        // loss += fabs(input.label[i] * log(layers[layers.size() - 1].nodes[i].ac_value));
        // double tmpe = fabs(input.label[i] - layers[layers.size() - 1].nodes[i].ac_value);
        // loss += tmpe * tmpe / 2;

        //The only one you need to define.
        //output layer: softmax + cross entropy loss
        layers[layers.size() - 1].nodes[i].delta = layers[layers.size() - 1].nodes[i].ac_value - input.label[i];

        //output layer: sigmoid + cross entropy loss
        // layers[layers.size() - 1].nodes[i].delta = layers[layers.size() - 1].nodes[i].ac_value - input.label[i];

        //output layer: sigmoid + squared error loss
        // layers[layers.size() - 1].nodes[i].delta = (layers[layers.size() - 1].nodes[i].ac_value - input.label[i]) * (1 - layers[layers.size() - 1].nodes[i].ac_value) * layers[layers.size() - 1].nodes[i].ac_value;

        //output layer: relu + cross entropy loss
        // layers[layers.size() - 1].nodes[i].delta = -input.label[i] * layers[layers.size() - 1].activate_function_backward(layers[layers.size() - 1].nodes[i].ac_value) / layers[layers.size() - 1].nodes[i].ac_value + (1 - input.label[i]) * layers[layers.size() - 1].activate_function_backward(layers[layers.size() - 1].nodes[i].ac_value) / (1 - layers[layers.size() - 1].nodes[i].ac_value);

        //output layer: relu + squared error loss
        // layers[layers.size() - 1].nodes[i].delta = (layers[layers.size() - 1].nodes[i].ac_value - input.label[i]) * backward_relu(layers[layers.size() - 1].nodes[i].ac_value);
    }

    // remained layers
    for (int k = int(layers.size()) - 2; k >= 0; --k)
    {
        for (int i = 0; i < layers[k].nodes_num; ++i)
        {
            double sum = 0;
            for (int j = 0; j < layers[k + 1].nodes_num; ++j)
            {
                sum += layers[k + 1].nodes[j].delta * layers[k + 1].nodes[j].weight[i];
            }
            layers[k].nodes[i].delta = sum * layers[k].ac_fun_backward(layers[k].nodes[i].in_value);
        }
    }

    /* 
        updata w_gradient,b_gradient
    */

    //first hidden layer
    for (int i = 0; i < layers[0].nodes_num; ++i)
    {
        layers[0].nodes[i].b_gradient += layers[0].nodes[i].delta;
        for (int j = 0; j < layers[0].input_num; ++j)
        {
            layers[0].nodes[i].w_gradient[j] += layers[0].nodes[i].delta * input.data[j];
        }
    }

    //remained layer
    for (size_t k = 1; k < layers.size(); ++k)
    {
        for (int i = 0; i < layers[k].nodes_num; ++i)
        {
            layers[k].nodes[i].b_gradient += layers[k].nodes[i].delta;
            for (int j = 0; j < layers[k - 1].nodes_num; ++j)
            {
                layers[k].nodes[i].w_gradient[j] += layers[k].nodes[i].delta * layers[k - 1].nodes[j].ac_value;
            }
        }
    }
}

//update weight&bias
void Network::step()
{
    for (size_t k = 0; k < layers.size(); ++k)
    {
        for (int i = 0; i < layers[k].nodes_num; ++i)
        {
            layers[k].nodes[i].bias -= lr * layers[k].nodes[i].b_gradient / step_len;
            for (int j = 0; j < layers[k].input_num; ++j)
            {
                layers[k].nodes[i].weight[j] -= lr * layers[k].nodes[i].w_gradient[j] / step_len;
            }
        }
    }
}

void Network::weight_init()
{
    for (size_t k = 0; k < layers.size(); ++k)
    {
        layers[k]._layer_weight_init(layers[k].input_num);
    }
}

void Network::zero_grad()
{
    step_len = 0;
    for (size_t k = 0; k < layers.size(); ++k)
    {
        layers[k]._layer_zero_grad();
    }
}

void Network::set_lr(double learning_rate)
{
    lr = learning_rate;
}

void Network::predict(vector<Input> &test_group)
{
    int num = test_group.size();
    cout << "prediction:" << endl;
    for (int i = 0; i < num; ++i)
    {
        forward(test_group[i]);

        cout << "test_data:";
        for (double &value : test_group[i].data)
            cout << value << "\t";

        cout << "\tlabel:";
        for (double &value : test_group[i].label)
            cout << value << "\t";

        vector<double> vec_pre;
        for (int i = 0; i < layers[layers.size() - 1].nodes_num; ++i)
        {
            vec_pre.emplace_back(layers[layers.size() - 1].nodes[i].ac_value);
        }
        cout << "\tpredict: ";
        for (double &value : vec_pre)
            cout << value << "\t";
        cout << endl;
    }
}

void Network::predict_minist(vector<Input> &test_group)
{
    int num = test_group.size();
    cout << "prediction:" << endl;
    int right_ans = 0;
    for (int i = 0; i < num; ++i)
    {

        // cout << "label: ";
        int label = 0;
        for (int j = 0; j < 10; ++j)
        {
            if (test_group[i].label[j] == 1)
            {
                // cout << j << " ";
                label = j;
                break;
            }
        }

        forward(test_group[i]);
        int max_arg = 0;
        double max = 0;
        for (int j = 0; j < layers[layers.size() - 1].nodes_num; ++j)
        {
            // cout << layers[layers.size() - 1].nodes[j].ac_value << " ";
            if (layers[layers.size() - 1].nodes[j].ac_value > max)
            {
                max = layers[layers.size() - 1].nodes[j].ac_value;
                max_arg = j;
            }
        }
        // cout << "\tpre: " << max_arg << endl;

        if (label == max_arg)
            ++right_ans;
    }
    cout << "acc: " << double(right_ans) / double(num) << endl;
}