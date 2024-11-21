import numpy as np
from manim import *

def F1(x,k,m,lb):
    def z(x,k,m,lb):
        cond= np.abs(x)/k - np.floor(np.abs(x)/k)
        return [1-m+(m/lb)*i if i<lb else 1-m+(m/(1-lb))*(1-i) for i in cond]
    c=z(x,k,m,lb)
    p=(x-40)*(x-185)*x*(x+50)*(x+180)
    return 3e-9*np.abs(p)*c+10*np.abs(np.sin(0.1*x))

def bayes_opt_step(self, cfg, context):
    # First slide, draw function to optimize
    F1 = context.get("F1")
    graphs = VGroup()
    grid = Axes(x_range=[-200, 200, 20], y_range=[0, 200, 20],
                x_length=10, y_length=5, tips=False,
                axis_config={"color": BLACK})
    graphs += grid
    graphs += Text(r"Objective Function to minimize", font_size=self.s_size).next_to(grid, UP)
    graphs += Text(r"Opt. param.", font_size=self.s_size).next_to(grid, DOWN+RIGHT).shift(LEFT)
    graphs += grid.plot(lambda x: F1(np.array([x]), 1, 0, 0.01)[0], color=self.main_color)
    self.play(
        FadeIn(graphs[:-1], run_time=self.fadein_rt),
        DrawBorderThenFill(graphs[-1], run_time=3*self.drawborderthenfill_rt)
    )
    self.last = graphs
    self.next_slide()  # Waits user to press continue to go to the next slide

    # Start BO algorithm
    from numpy.random import RandomState
    from bayes_opt import BayesianOptimization
    from bayes_opt import UtilityFunction
    N = int(cfg.num_samples)
    acquisition_function = cfg.acquisition_function
    kappa = float(cfg.kappa)
    def obj_func(x):
        return -F1(np.array([x]), 1,0,0.01)[0]
    def posterior(optimizer, x, y, X):
        optimizer._gp.fit(x, y)
        mu, sigma = optimizer._gp.predict(X, return_std=True)
        return (mu, sigma)
    optimizer = BayesianOptimization(obj_func, {'x': (-200, 200)}, random_state=100)
    acq_function = UtilityFunction(kind="ei", kappa=kappa)
    optimizer.maximize(init_points=N, n_iter=0, acquisition_function=acq_function)

    # slide 1 initial state
    xx = np.array([[res["params"]["x"]] for res in optimizer.res])
    yy = np.array([res["target"] for res in optimizer.res])
    sample = [e[0] for e in xx]# [xx[0][0], xx[1][0]]
    sample_dots = [Dot(color=self.secondary_color,fill_opacity=1.0).move_to(grid.c2p(e, 0)) for e in sample]
    lbl = Text("0. Guess initial samples (SOBOL)", font_size=self.s_size).next_to(grid, 0.5*DOWN)
    self.play(*[FadeIn(d, run_time=self.fadein_rt) for d in sample_dots],
              FadeIn(lbl, run_time=self.fadein_rt))
    self.next_slide()

    lbl_position = 3.5*RIGHT+1.5*UP

    # slide 1 evaluate objective function
    sample_f = [-obj_func(e) for e in sample]
    dot_anims = [sample_dots[i].animate.move_to(grid.c2p(sample[i], sample_f[i])) for i in range(len(sample))]
    self.play(AnimationGroup(*dot_anims),
              Transform(lbl, Text("1. Evaluate objective func.", font_size=self.s_size).move_to(lbl_position), run_time=self.transform_rt))
    self.next_slide()

    # slide 2 fit gaussian process
    gp = grid.plot(lambda x: -posterior(optimizer, xx, yy, np.array([x]).reshape(1, -1))[0][0], color=self.warn_color)
    graphs.add(gp)
    self.play(DrawBorderThenFill(gp, run_time=3*self.drawborderthenfill_rt),
              Transform(lbl, Text("2. Fit a Gaussian process model", font_size=self.s_size).move_to(lbl_position), run_time=self.transform_rt))
    self.next_slide()

    # slide 3 confidence
    def confidence(optimizer, x, y, X, interval):
        mu, s = posterior(optimizer, x, y, np.array([X]).reshape(1, -1))
        return -(mu[0] + interval*s[0])
    ce11 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, 1.95), color=self.warn_color)
    ce12 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, -1.95), color=self.warn_color)
    ce1 = grid.get_area(ce11, bounded_graph=ce12, opacity=0.5, x_range=(-200, 200))
    self.play(DrawBorderThenFill(ce1),
              Transform(lbl, Text("with a 95% confidence interval", font_size=self.s_size).move_to(lbl_position)))
    self.next_slide()

    # slide 4 acquisition function
    ei = grid.plot(lambda x: 100+5*acq_function.utility([[x]], optimizer._gp, 0)[0], color=self.important_color)
    self.play(FadeOut(ce1, gp, run_time=self.fadeout_rt))
    ei_area = grid.get_area(ei, x_range=(-200, 200), opacity=0.3, color=[self.important_color, self.important_color])
    self.play(FadeIn(ei_area, run_time=self.fadein_rt),
              Transform(lbl, Text("3. Estimate acquisition function (eg. EI)", font_size=self.s_size).move_to(lbl_position)))
    self.next_slide()

    niters = int(cfg.n_iters)
    for i in range(niters):
        optimizer.maximize(init_points=0, n_iter=1)
        xx = np.array([[res["params"]["x"]] for res in optimizer.res])
        yy = np.array([res["target"] for res in optimizer.res])
        sample = [e[0] for e in xx]
        sample_dots.append(Dot(color=self.secondary_color).move_to(grid.c2p(sample[-1], 0)))
        sample_f = [-obj_func(e) for e in sample]
        if i<4:
            # visualize first three iters
            if i == 0:
                self.play(FadeOut(ei_area, run_time=self.fadeout_rt))
            else:
                self.play(FadeOut(ei_area, gp, run_time=self.fadeout_rt))
            graphs.add(sample_dots[-1])
            iter_text = f"Iteration {i}"
            if i==0:
                iter_text = f"Pick next sample maximizing EI"
            anims = [d.animate.set_color(self.main_color) for d in sample_dots[:-1]]
            self.play(AnimationGroup(*anims),
                sample_dots[-1].animate.move_to(grid.c2p(sample[-1], sample_f[-1])),
                Transform(lbl, Text(iter_text, font_size=self.s_size).move_to(lbl_position)))
            self.next_slide()
            gp = grid.plot(lambda x: -posterior(optimizer, xx, yy, np.array([x]).reshape(1, -1))[0][0], color=self.warn_color)
            graphs.add(gp)
            self.play(DrawBorderThenFill(gp, run_time=self.drawborderthenfill_rt))
            self.next_slide()
            ce11 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, 1.96), color=self.warn_color)
            ce12 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, -1.96), color=self.warn_color)
            ce1 = grid.get_area(ce11, bounded_graph=ce12, opacity=0.5, x_range=(-200, 200))
            self.play(DrawBorderThenFill(ce1, run_time=self.drawborderthenfill_rt))
            self.next_slide()
            ei = grid.plot(lambda x: 100+5*acq_function.utility([[x]], optimizer._gp, 0)[0], color=self.warn_color)
            ei_area = grid.get_area(ei, x_range=(-200, 200), opacity=0.3, color=[self.important_color, self.important_color])
            self.play(FadeOut(ce1, run_time=self.fadeout_rt), FadeIn(ei_area, run_time=self.fadein_rt))
            self.next_slide()
        else:
            if i == niters-1:
                self.play(FadeOut(ei_area, gp, run_time=self.fadeout_rt),
                    Transform(lbl, Text("After some iterations...", font_size=self.s_size).move_to(lbl_position), run_time=self.transform_rt))

                graphs.add(*sample_dots[4:])
                anims = [d.animate.set_color(self.main_color) for d in sample_dots[:3]]
                self.play(AnimationGroup(*anims))
                anims = [sample_dots[j].animate.move_to(grid.c2p(sample[j], sample_f[j])) for j in range(4, len(sample_dots))]
                self.play(AnimationGroup(*anims))
                gp = grid.plot(lambda x: -posterior(optimizer, xx, yy, np.array([x]).reshape(1, -1))[0][0], color=self.warn_color)
                graphs.add(gp)
                self.play(DrawBorderThenFill(gp, run_time=self.drawborderthenfill_rt))

                self.next_slide()
                ce11 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, 1.95), color=self.warn_color)
                ce12 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, -1.95), color=self.warn_color)
                ce1 = grid.get_area(ce11, bounded_graph=ce12, opacity=0.5, x_range=(-200, 200))
                self.play(DrawBorderThenFill(ce1, run_time=self.drawborderthenfill_rt))

                self.next_slide()
                ei = grid.plot(lambda x: 100+5*acq_function.utility([[x]], optimizer._gp, 0)[0], color=self.warn_color)
                ei_area = grid.get_area(ei, x_range=(-200, 200), opacity=0.3, color=[self.important_color, self.important_color])
                self.play(FadeIn(ei_area, run_time=self.fadein_rt))
        self.next_slide()
    
    # code configuration
    self.reset_step(cfg, self.layout[0])
    self.text_step(cfg.benchmark, self.last)
    self.code_step(cfg.parameters, self.last)
    self.code_step(cfg.objectives, self.last)
