#ifndef __MEX_OUTPUT_H__
#define __MEX_OUTPUT_H__

#include "mex.h"

#include "Eigen/Core"
#include <vector>

namespace MEX {
    class Output {
    public:
        Output(int size, mxArray** data);

        int size() const;

        bool has(int i) const;

        class Setter {
            public:
                Setter(mxArray** data);

                Setter& operator=(const Eigen::MatrixXd& output);

                // Setter& operator=(const Eigen::MatrixXb& output);

                Setter& operator=(const Eigen::ArrayXXd& output);

                Setter& operator=(const Eigen::ArrayXXi& output);

                Setter& operator=(const std::vector<Eigen::MatrixXd>& output);

                Setter& operator=(const std::string& s);

                Setter& operator=(const mxArray* a);

                Setter& operator=(const Setter& s);

                Setter& operator=(const double& d);

                Setter& operator=(const int& i);

                Setter& operator=(const bool& b);

            private:
                mxArray** mData;
        };

        MEX::Output::Setter operator[](int index) const;

    protected:
        Output();

        int mSize;
        mxArray** mData;

    private:
        //ToDo: Disable copy operation to avoid confusion?!
        //Output(const Output& other);
        //Output & operator=(const Output&);
    };
}

#endif
