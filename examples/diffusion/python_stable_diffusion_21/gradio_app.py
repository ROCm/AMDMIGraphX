#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

from txt2img import StableDiffusionMGX
import gradio as gr


def main():
    # Note: This will load the models, which can take several minutes
    sd = StableDiffusionMGX()

    def gr_wrapper(prompt, negative_prompt, steps, seed, scale):
        result = sd.run(str(prompt), str(negative_prompt), int(steps),
                        int(seed), float(scale))
        return StableDiffusionMGX.convert_to_rgb_image(result)

    demo = gr.Interface(
        gr_wrapper,
        [
            gr.Textbox(value="a photograph of an astronaut riding a horse",
                       label="Prompt"),
            gr.Textbox(value="", label="Negative prompt (Optional)"),
            gr.Slider(1, 100, step=1, value=20, label="Number of steps"),
            gr.Textbox(value=13, label="Random seed"),
            gr.Slider(1, 20, step=0.1, value=7.0, label="Guidance scale"),
        ],
        "image",
    )
    demo.launch()


if __name__ == "__main__":
    main()
