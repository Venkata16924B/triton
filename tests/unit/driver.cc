#include "triton/driver/backend.h"
#include "triton/driver/module.h"
#include "triton/driver/kernel.h"
#include "triton/driver/stream.h"
#include "lodepng.h"

namespace drv = triton::driver;

int main() {
  static const int W = 3200;
  static const int H = 2400;
  drv::context* ctx = drv::backend::contexts::get_default();
//  drv::device* device = ctx->device();
  drv::stream* stream = drv::stream::create(ctx);
  drv::vk_module module(ctx, nullptr);
  drv::vk_kernel kernel(&module, "main");
  drv::vk_buffer buffer(ctx, W*H*16);
  kernel.setArg(0, &buffer);
  stream->enqueue(&kernel, {W/32, H/32, 1}, {32, 32, 1});
  stream->synchronize();

  struct Pixel {
      float r, g, b, a;
  };
  std::vector<Pixel> data(W*H);
  stream->read(&buffer, false, 0, buffer.size(), data.data());

  // Get the color data from the buffer, and cast it to bytes.
  // We save the data to a vector.
  std::vector<unsigned char> image;
  image.reserve(W * H * 4);
  for (int i = 0; i < W * H; i += 1) {
      image.push_back((unsigned char)(255.0f * (data[i].r)));
      image.push_back((unsigned char)(255.0f * (data[i].g)));
      image.push_back((unsigned char)(255.0f * (data[i].b)));
      image.push_back((unsigned char)(255.0f * (data[i].a)));
  }

  // Now we save the acquired color data to a .png.
  unsigned error = lodepng::encode("mandelbrot.png", image, W, H);
  if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
}
