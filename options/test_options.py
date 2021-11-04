from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        
        # additional options shared with train_options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        parser.add_argument('--display_ncols', type=int, default=4,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='vanilla',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        
        
        # test-specific options
        parser.add_argument('--no_html', action='store_true',
                            help='do not save test results to [opt.checkpoints_dir]/[opt.name]/web/')

        parser.add_argument('--show_num', type=int, default=10,
                            help='default number of test images to display in HTML file.')

        parser.add_argument('--show_by_ratio', action='store_true',
                            help='show a certain ratio of the test images in HTML file.')

        parser.add_argument('--show_ratio', type=float, default=0.1,
                            help='if show_by_ratio, then show this ratio of the test images in HTML file.')

        self.isTrain = False
        return parser
