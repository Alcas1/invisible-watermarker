# Art Shield Invisible Watermarker

#### This is the code we're utilizing in https://artshield.io to apply our invisible watermark
As we have been moving very fast on this project, the code may not be the most clean, but hopefully this will provide some extra clarity and transparency

#### We referenced this library for the algorithm: https://github.com/ShieldMnt/invisible-watermark
#### We've made slight changes to use the Pillow library instead of OpenCV2, and we compare if either saving as JPEG or PNG results in a successful watermarking.

Some extra code is included for Vercel serverless integration(AWS lambda under the hood)
Nothing fancy, only some code around parallel processing(used to decode JPG and PNG at once) that AWS lacks by default
