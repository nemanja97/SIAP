import mimetypes
import os
import shutil
import uuid
from os.path import join

import discord
import urllib3
from discord.ext import commands

from evaluate import load_vocabulary, load_model, caption_image
from utils.CheckpointUtils import load_checkpoint
from utils.Constants import DISCORD_TOKEN, TEMP_PATH
from utils.ImageTransormation import transform


def is_url_image(url):
    mimetype, encoding = mimetypes.guess_type(url)
    return mimetype and mimetype.startswith('image')


def generate_image_path():
    return join(TEMP_PATH, str(uuid.uuid4()))


def prepare_for_labeling(image_path):
    image = transform(image_path).unsqueeze(0)
    os.remove(image_path)
    return image


class DiscordBot(discord.ext.commands.Bot):
    def __init__(self, prefix):
        discord.ext.commands.Bot.__init__(self, command_prefix=prefix)
        self.http_client = urllib3.PoolManager()
        self.checkpoint = load_checkpoint('model.tar', 'cpu')
        self.vocabulary = load_vocabulary(self.checkpoint)
        self.model = load_model(self.checkpoint, self.vocabulary)
        self.model.eval()
        self.add_commands()

    def get_image(self, image_url):
        path = generate_image_path()

        with self.http_client.request('GET', image_url, preload_content=False) as response, open(path,
                                                                                                 'wb') as output_file:
            shutil.copyfileobj(response, output_file)

        response.release_conn()

        return path

    def do_labeling(self, image):
        return caption_image(image, self.model, self.vocabulary)

    def download_and_label(self, image_url):
        if is_url_image(image_url):
            return self.do_labeling(prepare_for_labeling(self.get_image(image_url)))
        raise discord.DiscordException

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord!')

    def add_commands(self):
        @self.command(name='label', help='Gives image description', pass_context=True)
        async def label_image(ctx, image_url):
            await ctx.channel.send(self.download_and_label(image_url))

    async def on_command_error(self, ctx, _error):
        await ctx.channel.send('Invalid command argument')


if __name__ == '__main__':
    bot = DiscordBot('$')
    bot.run(DISCORD_TOKEN)
