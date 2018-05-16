let processor = {
  blurKernel: {},
  blurGPUKernel: null,
  mainImage: null,
  doubleBuffer: null,
  gpu: new GPU(),
  timerCallback: function() {
    if (this.video.paused || this.video.ended) {
      return;
    }
    this.computeFrame();
    let self = this;
    setTimeout(function () {
        self.timerCallback();
      }, 0);
  },

  doLoad: function() {
    this.video = document.getElementById("video");
    let self = this;
    this.video.addEventListener("play", function() {
        self.c1 = document.getElementById("c1");
        self.ctx1 = self.c1.getContext("2d");
        self.c2 = document.getElementById("c2");
        self.ctx2 = self.c2.getContext("2d");
        self.width = self.video.videoWidth;
        self.height = self.video.videoHeight;
        self.c1.width = self.width;
        self.c1.height = self.height;
        self.c2.width = self.width;
        self.c2.height = self.height;
        self.computeInit();
        self.timerCallback();
      }, false);
  },
  convolution: function (input, output, kernel) {
    let khalf = Math.floor(kernel.size / 2);
    let c;
    let p;
    let b = { x: 0, y: 0 };
    for (let x = khalf; x < input.x - khalf; x++) {
      for (let y = khalf; y < input.y - khalf; y++) {
        p = 0;
        c = 0;
        for (b.y = -khalf; b.y <= khalf; b.y++) {
          for (b.x = -khalf; b.x <= khalf; b.x++) {
            p += input.pixel[(y - b.y) * input.x + x - b.x] * kernel.matrix[c];
            c++;
          }
        }
        p = Math.floor(p);
        if (p > output.max) {
          p = output.max;
        }
        if ((y * input.x + x) % 1 != 0) {
          debugger;
        }
        output.pixel[y * input.x + x] = p;
      }
    }
  },
  convolution: function (input, output, kernel) {
    /*let c;
    let p;
    let b = { x: 0, y: 0 };
    for (let x = khalf; x < input.x - khalf; x++) {
      for (let y = khalf; y < input.y - khalf; y++) {
        p = 0;
        c = 0;
        for (b.y = -khalf; b.y <= khalf; b.y++) {
          for (b.x = -khalf; b.x <= khalf; b.x++) {
            p += input.pixel[(y - b.y) * input.x + x - b.x] * kernel.matrix[c];
            c++;
          }
        }
        p = Math.floor(p);
        if (p > output.max) {
          p = output.max;
        }
        if ((y * input.x + x) % 1 != 0) {
          debugger;
        }
        output.pixel[y * input.x + x] = p;
      }
    }*/
    var out = this.blurGPUKernel(GPU.input(input.pixel, [input.x, input.y]), GPU.input(kernel.matrix, [kernel.size, kernel.size]));
    for (let x = 0; x < output.x; x++) {
      for (let y = 0; y < output.y; y++) {
        output.pixel[x + output.x * y] = out[y][x];
      }
    }
  },
  blurCreateKernel: function (sigma) {
    let size = (2 * Math.floor(2 * sigma) + 3);
    let mean = Math.floor(size / 2);
    let kernel = {
      size: size,
      matrix: new Float32Array(size * size)
    };
    let c = 0;
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        kernel.matrix[c++] = Math.exp(-0.5 * (Math.pow((x - mean) / sigma, 2) + Math.pow((y - mean) / sigma, 2))) / (2 * Math.PI * sigma * sigma);
      }
    }
    this.blurKernel = kernel;

    this.blurGPUKernel = this.gpu.createKernel(function(a, b) {
      if (this.thread.x < this.constants.minxy || this.thread.x >= this.constants.maxx ||
          this.thread.y < this.constants.minxy || this.thread.y >= this.constants.maxy) {
        return (a[this.thread.y][this.thread.x]);
      }
      var p = 0;
      for (var ky = -this.constants.khalf; ky <= this.constants.khalf; ky++) {
        for (var kx = -this.constants.khalf; kx <= this.constants.khalf; kx++) {
          p += a[this.thread.y - ky][this.thread.x - kx] * b[ky][kx];
        }
      }
      return p;
    },
    {
      constants: {
        x: this.width,
        y: this.height,
        minxy: mean,
        maxx: this.width - mean,
        maxy: this.height - mean,
        khalf: mean,
      }
    }).setOutput([this.width, this.height]);
    this.blurGPUKernel.floatTextures = true;
  },
  blur: function (input, output) {
    this.convolution(input, output, this.blurKernel);
  },
  copyImageFormat: function (aIn, buff) {
    return {
      x: aIn.x,
      y: aIn.y,
      pixel: buff,
      max: aIn.max
    };
  },
  compute: function (image) {
    this.doubleBuffer = image.pixel;
    let doubleBuffer = this.copyImageFormat(image, this.doubleBuffer);

    this.blur(doubleBuffer, image);
  },
  computeInit: function() {
    this.blurCreateKernel(1.2);
    this.mainImage = new Float32Array(this.width * this.height);
    this.doubleBuffer = new Float32Array(this.width * this.height);
  },
  computeFrame: function() {
    this.ctx1.drawImage(this.video, 0, 0, this.width, this.height);
    let frame = this.ctx1.getImageData(0, 0, this.width, this.height);
    let l = frame.data.length / 4;
    let image = {
      x: this.width,
      y: this.height,
      pixel: this.mainImage,
      max: 16581375
    };
    let i;
    for (i = 0; i < l; i++) {
      let r = frame.data[i * 4 + 0];
      let g = frame.data[i * 4 + 1];
      let b = frame.data[i * 4 + 2];
      image.pixel[i] = ((r << 16) + (g << 8) + b);
    }
    this.compute(image);
    for (i = 0; i < l; i++) {
      let fpixel = image.pixel[i];
      let r = g = b = fpixel / (image.max / 255);
      frame.data[i * 4] = r;
      frame.data[i * 4 + 1] = g;
      frame.data[i * 4 + 2] = b;
      /*frame.data[i * 4 + 0] = fpixel >> 16 & 0xFF;
      frame.data[i * 4 + 1] = fpixel >> 8 & 0xFF;
      frame.data[i * 4 + 2] = fpixel & 0xFF;*/
    }
    this.ctx2.putImageData(frame, 0, 0);
    return;
  }
};
