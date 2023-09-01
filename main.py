import argparse
from argparse import RawTextHelpFormatter
import glob
from os import makedirs
from os.path import join, exists, basename, splitext
from tqdm import tqdm
import cv2
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from plyer import filechooser
import shutil
import os
from enhancement import enhance_image_exposure

Builder.load_string('''
<MainScreen>:
    orientation: 'vertical'
    spacing: 20
    padding: 20
    BoxLayout:
        orientation: 'horizontal'
        Button:
            text: 'Select Images'
            on_release: root.select_images()
            background_color: (1, 0, 1)
            color: (1, 1, 1, 1)
            font_size: '20sp'
            font_name: 'Arial' 
            size_hint: (0.5, 0.5) 
            width: 200
            height: 100 
            border: (4, 4, 4, 4) 
            border_radius: 10
            
        Button:
            text: 'Convert'
            on_release: root.convert_images()
            background_color: (1, 0, 1)
            color: (1, 1, 1, 1)
            font_size: '20sp'
            font_name: 'Arial' 
            size_hint: (0.5, 0.5) 
            width: 200
            height: 100 
            border: (4, 4, 4, 4) 
            border_radius: 10
            margin_top:-100
    ScrollView:
        GridLayout:
            id: grid
            cols: 3
            spacing: 20
            padding: 30
''')


class MainScreen(BoxLayout):
    def select_images(self):
        selected = filechooser.open_file(title='Select Images', filters=[('Image Files', '*.png;*.jpg;*.jpeg')])
        if selected:
            print(selected)
            src_file = selected[0]
            dst = os.path.dirname(os.path.abspath(__file__)) + '\\DTL_Application\\Normal'
            print(dst)
            shutil.copy(src_file, dst)

    path = os.path.join('DTL_Application', 'Normal')
    if not path:
        os.makedirs(path)

    # directory = join('data/Enhanced-Image')
    # if not exists(directory):
    #     os.makedirs(directory)
    #
    # directory = join('data/Enhanced-Image/Normal')
    # if not exists(directory):
    #     os.makedirs(directory)

    def convert_images(self):

        def main(args):

            # load images
            imdir = args.folder
            ext = ['png', 'jpg', 'bmp'] 
            files = []
            [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
            images = [cv2.imread(file) for file in files]

            # create save directory
            directory = join(imdir, "Enhanced_Image_Files")
            if not exists(directory):
                makedirs(directory)
            print(args)
            # enhance images
            for i, image in tqdm(enumerate(images), desc="Enhancing images"):
                enhanced_image = enhance_image_exposure(image, args.gamma, args.lambda_, not args.lime,
                                                        sigma=args.sigma, bc=args.bc, bs=args.bs, be=args.be,
                                                        eps=args.eps)
                print(i)
                filename = basename(files[i])
                name, ext = splitext(filename)
                method = "LIME" if args.lime else "DUAL"
                corrected_name = f"{name}_{method}_g{args.gamma}_l{args.lambda_}{ext}"
                cv2.imwrite(join(directory, corrected_name), enhanced_image)
                self.ids.grid.add_widget(
                    Image(source='DTL_Application\\Normal\\Enhanced_Image_Files\\' + corrected_name,
                          size_hint_y=None, height=100))

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(
                description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
                formatter_class=RawTextHelpFormatter
            )
            parser.add_argument("-f", '--folder',
                                default=os.path.dirname(os.path.abspath(__file__)) + '\\DTL_Application\\Normal\\',
                                type=str,
                                help="folder path to test images.")
            parser.add_argument("-g", '--gamma', default=0.6, type=float,
                                help="the gamma correction parameter.")
            parser.add_argument("-l", '--lambda_', default=0.15, type=float,
                                help="the weight for balancing the two terms in the illumination refinement optimization objective.")
            parser.add_argument("-ul", "--lime", action='store_true',
                                help="Use the LIME method. By default, the DUAL method is used.")
            parser.add_argument("-s", '--sigma', default=3, type=int,
                                help="Spatial standard deviation for spatial affinity based Gaussian weights.")
            parser.add_argument("-bc", default=1, type=float,
                                help="parameter for controlling the influence of Mertens's contrast measure.")
            parser.add_argument("-bs", default=1, type=float,
                                help="parameter for controlling the influence of Mertens's saturation measure.")
            parser.add_argument("-be", default=1, type=float,
                                help="parameter for controlling the influence of Mertens's well exposedness measure.")
            parser.add_argument("-eps", default=1e-3, type=float,
                                help="constant to avoid computation instability.")

            args = parser.parse_args()
            main(args)


class MyApp(App):
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    MyApp().run()
