import numpy as np
from manim import *

def brotato_step(self, cfg, context):

    self.image_step(cfg.icon, self.layout[0])
    self.items_step(cfg.description, self.layout[0])
    self.items_step(cfg.objectives, self.last)
    self.items_step(cfg.conventions, self.last)
    self.reset_step(cfg, self.last)
    import math
    def compute_desire(wave, stat, df, trial):
        """ported from the GDscript variant"""
        desired_wave_col = f'{stat}_desired_wave'
        undesired_lower_col = f'{stat}_undesired_lower'
        desire_aggressiveness_col = f'{stat}_desire_aggr'
        desire_degressiveness_col = f'{stat}_desire_degr'
        if not all(col in df.columns for col in [desired_wave_col, undesired_lower_col, desire_aggressiveness_col, desire_degressiveness_col]):
            return 0.5
        desired_wave = float(df[desired_wave_col].values[trial])
        undesired_lower = float(df[undesired_lower_col].values[trial])
        desire_aggressiveness = float(df[desire_aggressiveness_col].values[trial])
        desire_degressiveness = float(df[desire_degressiveness_col].values[trial])
        if wave <= undesired_lower:
            return 0.0
        if wave <= desired_wave:
            return math.exp(-pow(wave - desired_wave, 2) / (2 * pow(desire_aggressiveness, 2)))
        else:
            return math.exp(-pow(wave - desired_wave, 2) / (2 * pow(desire_degressiveness, 2)))
    data = {
        'stat_desired_wave': [11], 
        'stat_undesired_lower': [6], 
        'stat_desire_aggr': [20], 
        'stat_desire_degr': [3]
    }
    import pandas as pd
    df = pd.DataFrame(data)
    axes = Axes(x_range=[1, 20, 1], y_range=[0.0, 1.0, 0.2],
                x_length=10, y_length=5, tips=False,
                axis_config={"color": BLACK},)
    x_tick_labels = [1, 20]
    y_tick_labels = [0, 1]
    axes.x_axis.set_tick_labels(x_tick_labels)
    axes.y_axis.set_tick_labels(y_tick_labels)
    axes_labels = Group(
        Text(r"Wave", font_size=self.s_size).next_to(axes, DOWN, buff=0.4),
        Text(r"Desire for action", font_size=self.s_size).next_to(axes, LEFT).rotate(PI/2).shift(0.2*RIGHT),
        Text("20", font_size=self.s_size).move_to(axes.c2p(20, 0)).shift(0.4*DOWN),
        Text("1", font_size=self.s_size).move_to(axes.c2p(1, 0)).shift(0.4*DOWN),
        Text("0", font_size=self.s_size).move_to(axes.c2p(0, 0)).shift(0.2*LEFT),
        Text("1", font_size=self.s_size).move_to(axes.c2p(0, 1)).shift(0.2*LEFT),
    )
    def plot_graph(stat, df, trial=0, color=self.main_color):
        undesired_lower = float(df[f'{stat}_undesired_lower'].values[trial])
        threshold=0.1
        x0 = undesired_lower-threshold
        x1 = undesired_lower+threshold
        def exp_function(wave):
            return compute_desire(wave, stat, df, trial)
        plot1 = axes.plot(
            lambda _: 0.0,
            x_range=[1, x0],
            color=color
        )
        plot1_2 = axes.plot(
            lambda wave: exp_function(x1)/(x1-x0)*(wave-x0),
            x_range=[x0,x1],
            color=color
        )
        if undesired_lower <= 1:
            x0 = 1
            x1 = 1
        plot2 = axes.plot(
            lambda wave: exp_function(wave),
            x_range=[x1, 20],
            color=color
        )
        if undesired_lower <= 1:
            return VGroup(plot2)
        return VGroup(plot1, plot1_2, plot2)
    graph = plot_graph("stat", df)
    self.play(FadeIn(axes, axes_labels, graph, run_time=self.fadein_rt))
    dots = Group(
        Dot(color=self.warn_color).move_to(axes.c2p(df["stat_desired_wave"][0], 0)),
        Dot(color=self.important_color).move_to(axes.c2p(df["stat_undesired_lower"][0], 0))
    )
    dots_labels = Group(
        Text("desired_wave", font_size=self.s_size, color=self.warn_color).next_to(dots[0], DOWN, buff=0.1),
        Text("undesired_lower", font_size=self.s_size, color=self.important_color).next_to(dots[1], DOWN, buff=0.1),
    )
    self.play(
        *[FadeIn(i, run_time=self.fadein_rt) for i in dots],
        *[FadeIn(i, run_time=self.fadein_rt) for i in dots_labels],
    )

    new_data = {
        'stat_desired_wave': [15], 
        'stat_undesired_lower': [4], 
        'stat_desire_aggr': [3], 
        'stat_desire_degr': [2]
    }
    def update_graph(mob, stat, df, trial=0, color=self.main_color):
        graph = plot_graph(stat, df, trial, color)
        mob.become(graph)
    for item in new_data.items():
        data[item[0]] = item[1]
        df = pd.DataFrame(data)
        self.next_slide()
        anims = []
        if item[0] == "stat_desired_wave":
            anims.append(dots[0].animate.move_to(axes.c2p(item[1][0], 0)))
            anims.append(MaintainPositionRelativeTo(dots_labels[0], dots[0]))
        elif item[0] == "stat_undesired_lower":
            anims.append(dots[1].animate.move_to(axes.c2p(item[1][0], 0)))
            anims.append(MaintainPositionRelativeTo(dots_labels[1], dots[1]))
        elif item[0] == "stat_desire_aggr":
            anims.append(FadeIn(Text("aggression", font_size=self.s_size).move_to(axes.c2p(7, 0.5)), run_time=self.fadein_rt))
        else:
            anims.append(FadeIn(Text("'degression'", font_size=self.s_size).move_to(axes.c2p(20, 0.5)), run_time=self.fadein_rt))
        self.play(
            *anims,
            UpdateFromFunc(graph, lambda mob: update_graph(mob, "stat", df))
        )

    self.reset_step(cfg, self.last)
    self.text_step(cfg.parameters, self.layout[0])
    self.items_step(cfg.parameter_analysis, self.last)
    self.reset_step(cfg, self.last)
    self.text_step(cfg.bro_objectives, self.layout[0])
    self.items_step(cfg.bro_objective_analysis, self.last)
    self.items_step(cfg.bro_foambo, self.last)
    self.reset_step(cfg, self.last)

    df = pd.read_csv("data/WellRounded_With_Stick_report.csv")
    TRIAL = 52
    economy = [ "harvesting", "luck", "engineering", "speed", "range" ]
    survival = [ "armor", "dodge", "hp_regeneration", "lifesteal", "max_hp", "speed" ]
    damage = [ "attack_speed", "crit_chance", "elemental_damage", "melee_damage", "ranged_damage", "range", "percent_damage" ]
    colors = [
        self.main_color, self.secondary_color, self.warn_color, self.important_color,
        #color.GREEN_C, color.GRAY_D, color.TEAL_D
        "#018081", "#7e3b19", "#274c92"
    ]
    stats = {"Economy": economy, "Survival": survival, "Damage": damage}
    for category in stats:
        axes = Axes(x_range=[1, 20, 1], y_range=[0.0, 1.0, 0.2],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": BLACK},)
        x_tick_labels = [1, 20]
        y_tick_labels = [0, 1]
        axes.x_axis.set_tick_labels(x_tick_labels)
        axes.y_axis.set_tick_labels(y_tick_labels)
        axes_labels = Group(
            Text(r"Wave", font_size=self.s_size).next_to(axes, DOWN, buff=0.6),
            Text(f"{category} stats", font_size=self.s_size).next_to(axes, LEFT).rotate(PI/2).shift(0.2*RIGHT),
            Text("20", font_size=self.s_size).move_to(axes.c2p(20, 0)).shift(0.4*DOWN),
            Text("9", font_size=self.s_size).move_to(axes.c2p(9, 0)).shift(0.4*DOWN),
            Text("1", font_size=self.s_size).move_to(axes.c2p(1, 0)).shift(0.4*DOWN),
            Text("0", font_size=self.s_size).move_to(axes.c2p(0, 0)).shift(0.2*LEFT),
            Text("1", font_size=self.s_size).move_to(axes.c2p(0, 1)).shift(0.2*LEFT),
        )
        self.play(FadeIn(axes, axes_labels, run_time=self.fadein_rt))
        for i in range(len(stats[category])):
            stat = stats[category][i]
            graph = plot_graph(stat, df, trial=TRIAL, color=colors[i])
            legend = Text(stat, font_size=self.vs_size, color=colors[i]).move_to(axes.c2p(22, 1-i*0.05))
            self.next_slide()
            self.play(FadeIn(graph, legend, run_time=self.fadein_rt))
        self.reset_step(cfg, self.last)

    self.items_step(cfg.objective_results, self.layout[0])
