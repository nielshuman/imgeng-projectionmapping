import pyglet
pyglet.options["shadow_window"] = False

import arcade

class Test(arcade.Window):
    def __init__(self):
        super().__init__(800, 600, "Arcade Test")

    def on_draw(self):
        self.clear()
        arcade.draw_circle_filled(400, 300, 80, arcade.color.RED)
        arcade.draw_text("Arcade works!", 400, 200, arcade.color.WHITE, 24, anchor_x="center")

Test()
arcade.run()