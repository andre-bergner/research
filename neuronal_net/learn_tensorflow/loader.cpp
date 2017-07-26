// Two ways to compile:
// c++ loader.cpp -I  /usr/local/lib/python3.5/site-packages/tensorflow/include/ -ltensorflow
// c++ loader.cpp -I  /usr/local/lib/python2.7/site-packages/tensorflow/include/ -ltensorflow_cc -L .
// install_name_tool -change bazel-out/darwin_x86_64-opt/bin/tensorflow/libtensorflow_cc.so @executable_path/libtensorflow_cc.so loader27



#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include "tensorflow/cc/ops/const_op.h"


class Session
{
    std::unique_ptr<tensorflow::Session> session_ = nullptr;

public:

    struct Exception { tensorflow::string  error_message; };

    Session()
    {
        tensorflow::Session* p;
        auto status = NewSession(tensorflow::SessionOptions(), &p);
        if (not status.ok())
            throw Exception{status.ToString()};
        session_.reset(p);
    }

    ~Session()
    {
        if (session_) (void)session_->Close();
    }

    void create( tensorflow::GraphDef g ) {
        auto status = session_->Create(g);
        if (not status.ok())
            throw Exception{status.ToString()};
    }

    auto run(const std::vector<std::pair<tensorflow::string, tensorflow::Tensor> >& inputs,
             const std::vector<tensorflow::string>& output_tensor_names,
             const std::vector<tensorflow::string>& target_node_names)
    {
        std::vector<tensorflow::Tensor> outputs;

        auto status = session_->Run( inputs, output_tensor_names, target_node_names, &outputs );
        if (not status.ok())
            throw Exception{status.ToString()};

        return outputs;
    }

    auto* operator->() { return session_.get(); }
};





int main(int args_num, char* args[]) {

    using namespace tensorflow;

    try {
        // Initialize a tensorflow session
        ::Session session;

        // Read in the protobuf graph we exported
        // (The path seems to be relative to the cwd. Keep this in mind
        // when using `bazel run` since the cwd isn't where you call
        // `bazel run` but from inside a temp folder.)
        GraphDef graph_def;

        auto status = ReadBinaryProto(Env::Default(), "models/graph.pb", &graph_def);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return 1;
        }

        session.create(graph_def);      // Add the graph to the session

        // Setup inputs and outputs:

        auto A = ops::Const({ {3.f, 2.f}, {-1.f, 0.f}});

        // Our graph doesn't require any inputs, since it specifies default values,
        // but we'll change an input to demonstrate.
        Tensor a(DT_FLOAT, TensorShape());
        a.scalar<float>()() = 3.0;

        Tensor b(DT_FLOAT, TensorShape());
        b.scalar<float>()() = 2.0;

        std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "a", a },
            { "b", b },
        };

        // Run the session, evaluating our "c" operation from the graph
        auto outputs = session.run(inputs, {"c"}, {});

        // Grab the first output (we only evaluated one graph node: "c")
        // and convert the node to a scalar representation.
        auto output_c = outputs[0].scalar<float>();

        // (There are similar methods for vectors and matrices here:
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

        // Print the results
        std::cout << "---------------------------\n";
        std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
        std::cout << output_c() << "\n"; // 30

        // Free any resources used by the session
        return 0;
    }

    catch (::Session::Exception&)
    {
        return 1;
    }
}
