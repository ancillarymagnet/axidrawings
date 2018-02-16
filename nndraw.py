# NNDRAW
# STUFF TO DRAW WITH BY NN
# n@hardwork.party

# scriptOp.inputs[0] is input
# scriptOp.inputs[0].points is all points on input .prims is prims (which has center member!)
# point has a member P, position as tuple (x,y,z)
# poly[x] gets vertex x, which has points as members

import math
import numpy as np
import numpy.random as rand
import random
PI = math.pi
TWOPI = PI * 2

def copy(so, inp=0):
	so.copy(so.inputs[inp])

def line(so, numPoints=2):
	return so.appendPoly(numPoints, closed=False, addPoints=True)

def point(so):
	return so.appendPoly(1, closed=False, addPoints=True)

def line_fr_to(so, fr=(0,0,0), to=(1,1,0), numPoints=2):
	"""
	takes scriptOp, fr and to (as tdu.Positions), and a number of points
	"""
	l = line(so, numPoints)
	for i in range(numPoints):
		f = i/(numPoints-1)
		l[i].point.P = lerp_tuple(f, fr, to) 

def array_to_line(so, array):
	l = line(so, len(array)) # this might be the reason we have connections at the end? len(pts-1?) probably not
	for i, p in enumerate(array):
		l[i].point.P = (p[0], p[1], p[2])
	return l

def bezier(so, numPoints=4):
	so.appendBezier(numPoints, closed=False)

def bezier_fr_to(so, fr, to, numPoints=4):
	bezier(so, numPoints)
	l = len(so.prims)
	debug(l)
	for p in range(numPoints):
		f = p / numPoints
		so.prims[l-1].updateAnchor(p, lerp_tuple(f, fr, to))

def curve_frto(so, fr, to, numPoints=10, curvature=PI, variation=1., squiggly=False):
	l = line_frto(so, fr, to, numPoints)
	
	if squiggly:
		var = np.random.rand(numPoints, 2)
		var *= variation
	else:
	# this is more of a walking curve
		var = np.zeros((numPoints, 2))
		walkx = 0
		walky = 0
		for i in range(numPoints):
			walkx += np.random.normal(scale=variation)  
			walky += np.random.normal(scale=variation) 
			var[i,0] += walkx
			var[i,1] += walky

	for i in range(numPoints):
		fy = math.sin(i/numPoints * curvature + var[i,0])
		fx = math.cos(i/numPoints * curvature + var[i,1])
		l[i].point.P += (fx, fy, 0)
	return l


def lerp(f, start, stop):
	return (f * (stop-start)) + start

def lerp_tuple2(f, start, stop):
	start = tdu.Vector(start[0],start[1])
	stop = tdu.Vector(stop[0],stop[1])
	return (f * (stop-start) + start)

def lerp_tuple(f, start, stop):
	start = tdu.Position(start)
	stop = tdu.Position(stop)
	return tdu.Position(f * (stop-start) + start)

def sum_tuples(t0, t1):
	return tuple(map(sum, zip(t0, t1)))

def clampf(n, lo, hi): 
		return max(lo, min(n, hi))

def clamp_tuple(t, lo, hi):
	ct = []
	for i in range(3):
		ct.append(clampf(t[i], lo[i], hi[i]))
	return tuple(ct)

def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def debug_array_to_table(array, tableName):
	op(tableName).clear()
	for i, v in enumerate(array):
		op(tableName).appendRow(v)

def print_vtx_pos(p):
	print('p: {},{},{}'.format( 
		p.point.P.x,
		p.point.P.y,
		p.point.P.z))

def print_point(p):
	print('p: {},{},{}'.format( 
		p.P.x,
		p.P.y,
		p.P.z))

def line_to_line(so, l1, l2, numPoints=2):
	l1_start, l1_end = l1
	l2_start, l2_end = l2

	for i in range(numPoints):
		f = i/(numPoints)
		p1 = lerp_tuple(f, l1_start, l1_end)
		p2 = lerp_tuple(f, l2_start, l2_end)
		line_frto(so, p1, p2)


def array_to_line(so, array):
	l = line(so, len(array))
	for i, p in enumerate(array):
		l[i].point.P = tdu.Position(p)
	return l

def frac_line_to_line(so, l1, l2, numPoints=2, drop=0):
	l1_start, l1_end = l1
	l2_start, l2_end = l2

	for i in range(numPoints):
		f = i/(numPoints)
		p1 = lerp_tuple(f, l1_start, l1_end)
		p2 = lerp_tuple(f, l2_start, l2_end)
		fractured_line(so, p1, p2, drop)

def fractured_line(so, s, e, fillFactor):
	# NOT FUNCTIONAL
	# draw a line from (sx, sy) to (ex, ey) that is only partially
	# complete (leaving gaps along the way)

	# while f < 1:
	# pick an amount p of the line to draw or not draw
	# flip a coin to decide on draw or not
	# draw or don't
	# move the head p down the line
	# f += p
	# import math

	f = 0
	# l = math.hypot(ex - sx, ey - sy)
	while f < 1:
		# a fraction of the line to draw, from zero up to the whole line
		p = abs(random.gauss(fillFactor,.5))
		if p >= (1-f):
			p = (1-f)
		# to draw or not to draw
		if random.randrange(2):
			l = line(so, 2)
			l[0].point.P = lerp_tuple(f, s, e)
			l[1].point.P = lerp_tuple(f+p, s, e)
			# print('f: {}'.format(f))
			# print('p: {}'.format(p))
			# print('start: {}'.format(l[0].point.P))
			# print('end: {}'.format(l[1].point.P))
		f += p


def fractured_line_np(so, s, e, fillFactor, minLength=0.01, maxLength=None):
	# OUTLINE FOR NP REFACTOR
	# if I decide on a number of line segments beforehand, I know
	# where I have to put them
	# so the fill factor is independent of the fragmentation, or the
	# segment count

	# maybe somehow generate arrays of random values
	# one for the 'fill' and one for the 'empty'
	# where the max of each fillfactor and 1-fillfactor
	# then sprinkle their segments together?


	# ALTERNATIVELY, TRY THE OG METHOD BUT KEEP A COUNTER
	# OF FILLED VS UNFILLED
	# less general but could work


	# while we're not at the end yet
	# generate a segment between minLength and min(maxLength, fillFactor * len(e-s))
	# are we still using fractional length?

	# if maxLength = None:
	# 	maxLength = len()

	pass


def delaunay(so):
	# NOT FUNCTIONAL
	copy(so)
	for p in so.points:
		print_point(p)
	from scipy.spatial import Delaunay
	
	pass


#### DRAWINGS

def fractured_line_grid(so, x, y, linesPerSquare, fillFactor):
	XSCALE = 11.69
	YSCALE = 8.27
	for i in range(x):
		for j in range(y):
			box_corner_x = i * XSCALE
			box_corner_Y = j * YSCALE
			rot = random.randrange(4)
			print(rot)
			# line_pairs = 
			# fractured_line(so, )

def feathers(so):
	for i in range(200):
		l = line(so)
		l[0].point.P = (i*i*2,i*100,0)
		l[1].point.P = (i*100,i*i*.5,0)

	for j in range(200):
		l3 = line(so)
		l3[0].point.P = (100+j*200,j*j,0)


# draw a spiral to mimic a shaded circle
def circle_shaded(so, center=(0,0), rad=1, spirals=1, 
	res=200, clockwise=True, rotStart=0, draw=True):

	rads = np.linspace(0.0, rad, res-1)
	# prime the point array with the center point
	pts = np.array([[center[0], center[1], 0.]])

	theta = rotStart
	thetaStep = ((spirals * TWOPI) / (res-2))
	cw = -1
	if not clockwise:
		cw = 1
	for r in rads:
		x = r * math.cos(theta) + center[0]
		y = r * math.sin(theta) + center[1]
		pts = np.append(pts, [[x,y,0]], axis=0)
		theta += thetaStep * cw

	if draw:
		array_to_line(so, pts)
	return pts


def circle(so, center=(0,0), rad=1, completepct=1, res=40, rotStart=0, draw=True):

	completepct = tdu.clamp(completepct, 0., 1.)
	start = rotStart * TWOPI
	end = (completepct + rotStart) * TWOPI
	
	rang = np.linspace(start, end, res)
	ys = np.sin(rang) * rad + center[0]
	xs = np.cos(rang) * rad + center[1]

	coords = zip(ys, xs)
	pts = np.zeros((len(xs),3))
	for i, c in enumerate(coords):
		pts[[i]] = [c[0], c[1], 0]


	if draw:
		array_to_line(so, pts)
	return pts


def square(so, center=(0,0), size=1, rot=0):
	rectangle(so, center, (size,size), rot)


def rectangle(so, center=(0,0,0), size=(1,1), rot=0, draw=True):
	pts = []
	half_size = (size[0]/2, size[1]/2)
	# bottom left
	pts.append(center + tdu.Vector(-half_size[0], -half_size[1], 0.))
	# top left
	pts.append(center + tdu.Vector(-half_size[0], half_size[1], 0.))
	# top right
	pts.append(center + tdu.Vector(half_size[0], half_size[1], 0.))
	# bottom right
	pts.append(center + tdu.Vector(half_size[0], -half_size[1], 0.))
	# bottom left again
	pts.append(center + tdu.Vector(-half_size[0], -half_size[1], 0.))

	# ROTATION
	m = tdu.Matrix()
	m.rotate(0, 0, rot*(360/TWOPI), pivot=center)
	rot_pts = [m * p for p in pts]

	if draw:
		array_to_line(so, rot_pts)
	return rot_pts


## TOOLS

def perspective(so, vpoint=(0,0,0), camz=-5):
	for p in so.points:
		# the lower the value of z, the more we move the xy towards the vanish point
		f = tdu.remap(p.P[2], camz, -camz, 0., 1.)
		newPos = lerp_tuple(f, vpoint, p.P)
		p.P = (newPos[0], newPos[1], p.P[2])

def projection(so, cam):
	proj = cam.projection(1, 1)
	view = cam.worldTransform
	view.invert()

	for p in so.points:
		p.P = proj * view * p.P


def rand_delete_prims(so, d=0.5):
	# prims = so.prims
	# so.clear()
	destroyed_count = 0
	debug(len(so.prims))
	for p in so.prims:
		r = random.random()
		if r < d:
			destroyed_count += 1
			p.destroy()
	debug(destroyed_count)


def rand_delete_points(so, d=0.6):
	for p in so.points:
		r = random.random()
		if r > d:
			p.destroy()


## DRAWINGS

def stair_pattern(so, size=(10,10), stepSize=0.1, rot=0):
	frame_x = size[0]
	frame_y = size[1]

	def stair_line(start):
		x, y = start[0], start[1]
		a_line = np.array([[x,y,0.]])
		while x < frame_x and y < frame_y:
			x = np.clip(x+stepSize, 0, frame_x)
			a_line = np.append(a_line, [[x,y,0.]], axis=0)
			if x == frame_x:
				break
			y = np.clip(y+stepSize, 0, frame_y)
			a_line = np.append(a_line, [[x,y,0.]], axis=0)
		array_to_line(so, a_line)

	# create an array to hold our lines
	start_x = 0.
	start_y = 0.
	x = start_x
	y = start_y
	while x < frame_x:
		stair_line((x,start_y))
		x += stepSize*2
	while y < frame_y:
		stair_line((start_x,y))
		y += stepSize*2

def stair_pattern_connected(so, size=(10,10), stepSize=0.1, rot=0):
	frame_x = size[0]
	frame_y = size[1]

	def stair_line(start, up=True):
		# go up and to the right until you hit an extent
		# start should be a tdu.Position
		line = [tdu.Position(start)]
		x, y = start[0], start[1]

		while x < frame_x and y < frame_y:
			y = clampf(y+stepSize, 0, frame_y)
			line.append(tdu.Position(x,y,0.))
			if y >= frame_y:
				break
			x = clampf(x+stepSize, 0, frame_x)
			line.append(tdu.Position(x,y,0.))
		if up == False:
			# reverse line
			line.reverse()
			x, y = start[0], start[1]
		return line, tdu.Position(x,y,0.)

	start_x = 0.
	start_y = 0.
	x = start_x + stepSize * 2
	y = start_y 
	all_lines = []
	up = True
	while y < frame_y:
		sl, last = stair_line((start_x, y, 0.), up=up)
		all_lines.extend(sl)
		y += stepSize*2
		up = not up
	while x < frame_x:
		sl, last = stair_line((x, start_y, 0.), up=up)
		sl.extend(all_lines)
		all_lines = sl
		x += stepSize * 2
		up = not up

	array_to_line(so, all_lines)


def hairy_normals(so):
	""" make a line for every normal on the input geo """
	points = so.inputs[0].points
	for i in range( len(points) ) :
		pt = points[i]
		normal = tdu.Vector(pt.N[0], pt.N[1], pt.N[2])
		normal.normalize()
		end = normal + pt.P
		poly = curve_frto(so, pt.P, end, numPoints=50, curvature=.5, variation=.025)


def imitator(so, numLines, numPoints=100, variation_x=0.05, variation_y=0.01):
	frame_x = 11.69
	frame_y = 8.27

	# set up the first line array
	firstline_points = np.array([[0.,0.,0.]])
	x, y = 0, 0
	for i in range(1, numPoints):
		f = float(i) / (numPoints-1)
		y = lerp(f, 0, frame_y)
		firstline_points = np.append(firstline_points, [[x, y, 0.]], axis=0)
	# draw first line - close enough interpolation to the one stored in the array
	l = line_frto(so, (0,0,0), (0,frame_y,0), numPoints)

	# draw the rest
	recentline_points = firstline_points
	# currentline_points = np.array([[0.,0.,0.]])
	x_step = frame_x / numLines
	currentline_points = np.zeros((numPoints,3))
	for line in range(numLines):
		x_modulation = math.sin((line/(numLines-1)*PI)) * 1.15
		x_modulation = min(x_modulation, 1.0)
		# print(x_modulation)
		x_norm = line * x_step
		for i, p in enumerate(recentline_points):
			# we actually want to blend between the step off from the previous point
			# and where we actually should be
			y_norm = firstline_points[i][1]
			x_inh = p[0] + random.gauss(0., variation_x) + x_step
			y_inh = p[1] + random.gauss(0., variation_y)
			x = lerp(x_modulation, x_norm, x_inh)
			y = lerp(x_modulation, y_norm, y_inh)
			currentline_points[i] = [x,y,0.]
		array_to_line(so, currentline_points)
		recentline_points = currentline_points


def input_radial_imitator(so, noiseCHOP, numLines, rstep=1, variation_r=0.05):
	"""assumes input geometry contains origin"""
	so.copy(so.inputs[0])
	noise = op(str(noiseCHOP))

	numPoints = len(so.inputs[0].points)
	source = np.zeros((numPoints, 3))
	# print(source)
	for i, p in enumerate(so.inputs[0].points):
		source[i] = [p.P[0], p.P[1], p.P[2]]
	# print(len(source))

	prevLine = source
	currentLine = np.zeros((numPoints,3))
	for j in range(numLines):
		# vx, vy, vz = random.gauss(0., variation_r), random.gauss(0., variation_r), 0.
		vx, vy, vz = noise[0][j] * variation_r, noise[0][j+1] * variation_r, 0.
		for i, p in enumerate(prevLine):
			# get the vector from point away from origin
			v = tdu.Vector(p[0], p[1], p[2])
			v *= rstep
			# add variation but make sure the ends meet
			if i == 0 or i == (numPoints-1):
				v += tdu.Vector(vx,vy,vz)
			else:
				v += tdu.Vector(noise[i][j] * variation_r,
					noise[i+1][j] * variation_r,
					0.)
			# add rstep (plus some variability)
			# write the new point
			currentLine[i] = [v[0], v[1], v[2]]

		array_to_line(so, currentLine)
		prevLine = currentLine
		

# TODO - MAKE THIS CONCENT_RECTANGLES
# CHECK THAT COFFSET IS WITHIN range FOR ASPECT
def concent_squares(so, center=(0,0), size=1., numSquares=10, cOffset=(0.0,0.0) ):
	xoor = cOffset[0]<-0.5 or 0.5<cOffset[0]
	yoor = cOffset[1]<-0.5 or 0.5<cOffset[1]
	if(xoor or yoor):
		raise ValueError('cOffset values must be between -0.5 and 0.5')

	cOffsetV = tdu.Vector(cOffset[0], cOffset[1], 0.0)

	targOffsetV = cOffsetV * size
	centerV = tdu.Vector(center[0], center[1], 0.)
	targCenterV = centerV - targOffsetV

	for i in range(1, numSquares+1):
		f = float(i)/(numSquares)
		s = f*float(size)
		c = lerp_tuple(f, targCenterV, centerV)
		square(so, c, s, 0)
	

def concent_sq_grid(so, x, y, sqprsq=10):
	frame_x = 10.
	frame_y = 10.
	# frame = 10
	size = min(frame_y / y, frame_x / x)
	debug(size)
	for i in range(x):
		for j in range(y):
			fx = float(i)/x
			fy = float(j)/y
			c = (fx * frame_x, fy * frame_y)
			concent_squares(so, c, size, numSquares=sqprsq, cOffset=(fx-0.5,fy-0.5))
			

def dot_rep(so, x, y, gridscaleX, gridscaleY, thresh, chop):
	for i in range(x):
		for j in range(y):
			u = math.floor((i/x) * 1277) # edge detection makes garbage
			v = math.floor((j/y) * 720)
			val = chop[v][u]
			if val > thresh:
				cX = i * gridscaleX / x
				cY = j * gridscaleY / y
				cX += (rand.random() * 0.02)
				cY += (rand.random() * 0.02)
				cw = rand.random() > 0.5
				r = rand.random() * 360
				s = rand.random() * 3
				circle_shaded(so, center=(cX,cY), rad=val/45, spirals=s, 
					res=15, clockwise=cw, rotStart=r)
	
def circle_links(so, x, y, gridscaleX,gridscaleY, chop):
	for i in range(x):
		for j in range(y):
			u = math.floor((i/x) * 1280)
			v = math.floor((j/y) * 720)
			val = chop[v][u]
			if val > 0.0:
				cX = i * gridscaleX / x
				cY = j * gridscaleY / y
				circle(so, center=(cX, cY), rad=0.0395, completepct=val, res=20, rotStart=val)


def circle_path(so, numIterations, center=(0,0), rad=1, d=None):
	if numIterations == 0:
		return
	circle(so, center, rad, res=100)
	if d == None:
		#unassigned for first iteration
		d = random.uniform(0, TWOPI)
	# w = rad * random.uniform(0,2)
	w = rad * random.gauss(0.5, 0.5)
	# debug(d, center, rad, w)
	x = w * math.cos(d)
	y = w * math.sin(d)
	walk = (x,y)
	c = sum_tuples(walk, center)
	newRad = distance(c, center) - rad
	t = random.uniform(0, TWOPI)
	r = t + random.random()
	circle_path(so, numIterations-1, c, newRad, t)


def circle_path_grid(so, x=11, y=8, gridscaleX=11.69, gridscaleY=8.27, rad=1):
	for i in range(x):
		for j in range(y):
			cX = i * gridscaleX / x
			cY = j * gridscaleY / y
			circle_path(so, 40, center=(cX,cY), rad=rad)


def circle_straight_across(so, numLines=10):
	gap = PI/numLines
	for i in range(numLines):
		p = line(so)
		igap = i*gap
		p[0].point.P = tdu.Position(math.sin(igap),math.cos(igap), 0)
		p[1].point.P = tdu.Position(math.sin(igap+PI), math.cos(igap+PI), 0)


def circle_parallel_lines(so, numLines, start, drop):
	random.seed(1) #deliberately nondeterministic
	for i in range(1,numLines+1):
		if(random.random() > drop):
			gap = (PI / (numLines+1)) * i
			fr = (math.sin(gap+start), math.cos(gap+start), 0)
			to = (math.sin(TWOPI-gap+start), math.cos(TWOPI-gap+start), 0)
			line_frto(so, fr, to)


def saddle(so, numLines):
	l1 = ((0,0,0),(-10,10,0))
	l2 = ((10,10,0), (0,0,0))
	line_to_line(so, l1, l2, numLines)



def chop_squares(so, x, y, gridscaleX, gridscaleY, chop):
	for i in range(x):
		for j in range(y):
			u = math.floor((i/x) * chop.numChans)
			v = math.floor((j/y) * chop.numSamples)
			val = chop[u][v]
			cX = (i/(x-1))*gridscaleX
			cY = (j/(y-1))*gridscaleY
			center = tdu.Position(cX,cY, 0)
			square(so, center, size=val*0.1, rot=val*TWOPI)


def paul_grid(so, x, y, gridscaleX, gridscaleY, chop):
	ratio = gridscaleX/gridscaleY
	for i in range(x):
		for j in range(y): 
			u = math.floor((i/x) * chop.numChans)
			v = math.floor((j/y) * chop.numSamples)
			val = chop[v][u]
			if i == 0:
				val = 0
			if True:
				# print(v)
				cX = i / gridscaleX
				cY = j / gridscaleY
				# rectangle(so, (cX, cY), ((1.0-val) * 0.075 * ratio, (1.0-val) * 0.075), val * TWOPI)
				rectangle(so, (cX, cY), ((val) * 0.075 * ratio, (val) * 0.075), 0)


def spirals_fr_to(so, numSpirals=10, fr=(0,0,0), to=(1,0,0), rad=1):
	for i in range(numSpirals):
		f = i/numSpirals
		f *= TWOPI
		f += PI
		f = f % TWOPI
		f = math.cos(f)
		pos = lerp_tuple(f, fr, to)
		circle_shaded(so, pos, rad=f*rad, spirals=f, 
			res=200, clockwise=True, rotStart=f*TWOPI)


def bouncing_spirals(so, numSpirals=10, numLines=100, fr=(0,0,0), to=(1,0,0), rad=1):
	fr = tdu.Position(fr)
	to = tdu.Position(to)
	for l in range(numLines):
		fr += (0,-.1, 0)
		to += (0,-.1, 0)
		for i in range(numSpirals-l):
			f = i/numSpirals
			pos = lerp_tuple(f, fr, to)
			pos[1] += abs(math.sin(f*TWOPI))*5
			circle_shaded(so, pos, rad=0.05*(1-f), spirals=2, res=100)


def pointCyclone_old(so):
	z = -100
	radius = 0
	theta = 0

	while z< 100:
		x = radius * math.cos(theta)
		y = radius * math.sin(theta)
		p = scriptOp.appendPoint()
		p.P = (x,y,z)
		
		radius += .003
		theta += 1
		z = z + 0.01

def circle_circles(so, numCircles=6, circsPerCirc=10, innerRad=2, maxRad=10):

	rang = np.linspace(0, (numCircles-1)/(numCircles)*TWOPI, numCircles)
	ys = np.sin(rang) * innerRad
	xs = np.cos(rang) * innerRad

	coords = zip(ys, xs)
	pts = np.zeros((len(xs),3))
	for i, c in enumerate(coords):
		pts[[i]] = [c[0], c[1], 0]

	for p in pts:
		for i in range(circsPerCirc):
			f = float(i)/circsPerCirc
			circle(so, (p[0],p[1]), rad=f*maxRad, res=40)


