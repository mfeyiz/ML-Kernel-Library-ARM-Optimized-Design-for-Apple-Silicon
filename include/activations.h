#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

namespace hwml {

void relu_neon(float* X, int size);
void sigmoid_neon(float* X, int size);
void relu_mt(float* X, int size);
void sigmoid_mt(float* X, int size);

} // namespace hwml

#endif // ACTIVATIONS_H