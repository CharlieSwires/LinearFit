import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;


@SuppressWarnings("serial")
public class LinearFit extends JPanel{
	class Point3D{
		public double x,y,t;
	}
	static RandomGenerator rand = new JDKRandomGenerator();

	private static double NOISE_VALUE = 5.0;
	private static int SAMPLES = 5;
	List<Point3D> points;
	List<Point3D> ppoints;
	List<Point3D> vppoints;
	protected static boolean render = true;
	JLabel minxl= new JLabel("Noise");;
	JTextField minx= new JTextField("5",20);;
	JLabel maxxl = new JLabel("Scale");;
	JTextField maxx =new JTextField("3",20);
	JLabel sl = new JLabel("Samples");;
	JTextField s =new JTextField("5",20);
	JButton doRender= new JButton("Render");
	JPanel canvas = new JPanel();
	JFrame jfrm = new JFrame("Kalman Filter");
	static LinearFit charlie;
	LinearFit(){	
		setBorder(BorderFactory.createLineBorder(Color.BLACK, 4));
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		int height = getHeight();
		int width = getWidth();
		g.setColor(Color.RED);
		double max_x = Double.parseDouble(maxx.getText());
		for (Point3D point: points) {
			g.drawLine((int)Math.round(point.x*max_x-max_x/2.0)+width/2, (int)Math.round(point.y*max_x)+height/2, (int)Math.round(point.x*max_x+max_x/2.0)+width/2,(int)Math.round(point.y*max_x)+height/2);
			g.drawLine((int)Math.round(point.x*max_x)+width/2, (int)Math.round(point.y*max_x-max_x/2.0)+height/2, (int)Math.round(point.x*max_x)+width/2,(int)Math.round(point.y*max_x+max_x/2.0)+height/2);
		}
		g.setColor(Color.BLUE);
		for (Point3D ppoint: ppoints) {
			g.drawLine((int)Math.round(ppoint.x*max_x-max_x/2.0)+width/2, (int)Math.round(ppoint.y*max_x)+height/2, (int)Math.round(ppoint.x*max_x+max_x/2.0)+width/2,(int)Math.round(ppoint.y*max_x)+height/2);
			g.drawLine((int)Math.round(ppoint.x*max_x)+width/2, (int)Math.round(ppoint.y*max_x-max_x/2.0)+height/2, (int)Math.round(ppoint.x*max_x)+width/2,(int)Math.round(ppoint.y*max_x+max_x/2.0)+height/2);
		}


	}
	void doASquare() throws InterruptedException {
		double tempx,tempy;
		double tempt;
		points = new ArrayList<Point3D>();
		ppoints = new ArrayList<Point3D>();
		vppoints= new ArrayList<Point3D>();

		tempx = 0.0; tempy= 0.0;
		tempt = 0.0;
		for (int i=0; i < 20;i++) {
			tempx -= 2.0;
			tempt += 1.0;
			predictAndDraw(addNoise(tempx),addNoise(tempy),tempt);

		}		
		for (int i=0; i < 20;i++) {
			tempy -= 2.0;
			tempt += 1.0;
			predictAndDraw(addNoise(tempx),addNoise(tempy),tempt);

		}		
		for (int i=0; i < 20;i++) {
			tempx += 2.0;
			tempt += 1.0;
			predictAndDraw(addNoise(tempx),addNoise(tempy),tempt);

		}
		for (int i=0; i < 20;i++) {
			tempy += 2.0;
			tempt += 1.0;
			predictAndDraw(addNoise(tempx),addNoise(tempy),tempt);

		}		

	}
	//===========================================================================================================================
	/**
	 * A Kalman filter implemented using SimpleMatrix.  The code tends to be easier to
	 * read and write, but the performance is degraded due to excessive creation/destruction of
	 * memory and the use of more generic algorithms.  This also demonstrates how code can be
	 * seamlessly implemented using both SimpleMatrix and DMatrixRMaj.  This allows code
	 * to be quickly prototyped or to be written either by novices or experts.
	 *
	 * @author Peter Abeles
	 */
	public class KalmanFilterSimple{

	    // kinematics description
	    private SimpleMatrix F,Q,H;

	    // sytem state estimate
	    private SimpleMatrix x,P;

	    public void configure(DMatrixRMaj F, DMatrixRMaj Q, DMatrixRMaj H) {
	        this.F = new SimpleMatrix(F);
	        this.Q = new SimpleMatrix(Q);
	        this.H = new SimpleMatrix(H);
	    }

	    public void setState(DMatrixRMaj x, DMatrixRMaj P) {
	        this.x = new SimpleMatrix(x);
	        this.P = new SimpleMatrix(P);
	    }

	    public void predict() {
	        // x = F x
	    	this.x = F.mult(this.x);

	        // P = F P F' + Q
	    	this.P = F.mult(this.P).mult(F.transpose()).plus(Q);
	    }

	    public void update(DMatrixRMaj _z, DMatrixRMaj _R) {
	        // a fast way to make the matrices usable by SimpleMatrix
	        SimpleMatrix z = SimpleMatrix.wrap(_z);
	        SimpleMatrix R = SimpleMatrix.wrap(_R);

	        // y = z - H x
	        SimpleMatrix y = z.minus(H.mult(this.x));

	        // S = H P H' + R
	        SimpleMatrix S = H.mult(P).mult(H.transpose()).plus(R);

	        // K = PH'S^(-1)
	        SimpleMatrix K = P.mult(H.transpose().mult(S.invert()));

	        // x = x + Ky
	        this.x = this.x.plus(K.mult(y));

	        // P = (I-kH)P = P - KHP
	        this.P = this.P.minus(K.mult(H).mult(this.P));
	    }

	    public DMatrixRMaj getState() {
	        return this.x.getMatrix();
	    }

	    public DMatrixRMaj getCovariance() {
	        return this.P.getMatrix();
	    }
	}
	// discrete time interval
	
	double dt = 1d;
	// position measurement noise (meter)
	double measurementNoise = NOISE_VALUE;
	// acceleration noise (meter/sec^2)
	double accelNoise = 0.0d;
	private KalmanFilterSimple filterX;
	private KalmanFilterSimple filterY;
	DMatrixRMaj x1;
	DMatrixRMaj y1;
	DMatrixRMaj Ax;
	DMatrixRMaj Hx;
	DMatrixRMaj Ry;
	DMatrixRMaj Rx;
	private DMatrixRMaj P0y;
	private DMatrixRMaj P0x;
	private DMatrixRMaj Ay;
	private DMatrixRMaj Hy;
	DMatrixRMaj Qy;
	DMatrixRMaj Qx;
	private double vx;
	private double vy;
	class KalmanX {

		public KalmanX(double paramx, double paramvx) {
			// A = [ 1 dt ]
			//	     [ 0  1 ]
			Ax = new DMatrixRMaj(new double[][] { { 1, dt }, { 0, 1 } });
			// B = [ dt^2/2 ]
			//	     [ dt     ]
			//B = new Array2DRowRealMatrix(new double[][] { { Math.pow(dt, 2d) / 2d }, { dt } });
			// H = [ 1 0 ]
			Hx = new DMatrixRMaj(new double[][] { { 1d, 0d } });
			// x = [ 0 0 ]
			x1 = new DMatrixRMaj(new double[][] {{ paramx},{paramvx  }});

			SimpleMatrix tmp = new SimpleMatrix(new double[][] {
				{ Math.pow(dt, 4d) / 4d, Math.pow(dt, 3d) / 2d },
				{ Math.pow(dt, 3d) / 2d, Math.pow(dt, 2d) } });
			// Q = [ dt^4/4 dt^3/2 ]
			//	     [ dt^3/2 dt^2   ]
			Qx = tmp.scale(Math.pow(accelNoise, 2)).getDDRM();
			// P0 = [ 1 1 ]
			//	      [ 1 1 ]
			P0x = new DMatrixRMaj(new double[][] { { 1, 1 }, { 1, 1 } });
			// R = [ measurementNoise^2 ]
			Rx = new SimpleMatrix(new double[][] {{ Math.pow(measurementNoise, 2) }}).getDDRM();

			// constant control input, increase velocity by 0.1 m/s per cycle
			//u = new ArrayRealVector(new double[] { 0.0d });

			filterX = new KalmanFilterSimple();
		    filterX.configure(Ax, Qx, Hx); 

		    filterX.setState(x1, P0x);

		}
	}	
	class KalmanY {

		public KalmanY(double paramy, double paramvy) {
			// A = [ 1 dt ]
			//	     [ 0  1 ]
			Ay = new DMatrixRMaj(new double[][] { { 1, dt }, { 0, 1 } });
			// B = [ dt^2/2 ]
			//	     [ dt     ]
			//B = new Array2DRowRealMatrix(new double[][] { { Math.pow(dt, 2d) / 2d }, { dt } });
			// H = [ 1 0 ]
			Hy = new DMatrixRMaj(new double[][] { { 1d, 0d } });
			// y = [ 0 0 ]
			y1 = new DMatrixRMaj(new double[][] {{paramy },{paramvy }});

			SimpleMatrix tmp = new SimpleMatrix(new double[][] {
				{ Math.pow(dt, 4d) / 4d, Math.pow(dt, 3d) / 2d },
				{ Math.pow(dt, 3d) / 2d, Math.pow(dt, 2d) } });
			// Q = [ dt^4/4 dt^3/2 ]
			//	     [ dt^3/2 dt^2   ]
			Qy = tmp.scale(Math.pow(accelNoise, 2)).getDDRM();
			// P0 = [ 1 1 ]
			//	      [ 1 1 ]
			P0y = new DMatrixRMaj(new double[][] { { 1, 1 }, { 1, 1 } });
			// R = [ measurementNoise^2 ]
			Ry = new SimpleMatrix(new double[][] {{ Math.pow(measurementNoise, 2) }}).getDDRM();

			// constant control input, increase velocity by 0.1 m/s per cycle
			//u = new ArrayRealVector(new double[] { 0.0d });

			filterY = new KalmanFilterSimple();
		    filterY.configure(Ay, Qy, Hy); 

		    filterY.setState(y1, P0y);

		}
	}



	private void predictAndDraw(double x, double y, double t) {
		int fred = points.size() - 6;
		fred = (fred > 0)? fred: 0;
		Point3D point = new Point3D();
		point.x = x; point.y = y; point.t = t;
		points.add(point);
		vx = vy = 0.0;
		SimpleMatrix tmp = new SimpleMatrix(new double[][] {
			{ Math.pow(dt, 4d) / 4d, Math.pow(dt, 3d) / 2d },
			{ Math.pow(dt, 3d) / 2d, Math.pow(dt, 2d) } });
		Ax = new DMatrixRMaj(new double[][] { { 1, dt }, { 0, 1 } });
		Qx = tmp.scale(Math.pow(accelNoise, 2)).getDDRM();
		Hx = new DMatrixRMaj(new double[][] { { 1d, 0d } });
		Rx = new SimpleMatrix(new double[][] {{ Math.pow(measurementNoise, 2) }}).getDDRM();
		filterX.configure(Ax, Qx, Hx); 
		P0x = new DMatrixRMaj(new double[][] { { 1, 1 }, { 1, 1 } });
		//x1 = new DMatrixRMaj(new double[][] {{ points.get(fred).x }, { points.get(fred).t }});
		x1 = new DMatrixRMaj(new double[][] {{ points.get(fred).x }, { 0d }});//?
	    filterX.setState(x1, P0x);
		Ay = new DMatrixRMaj(new double[][] { { 1, dt }, { 0, 1 } });
		Qy = tmp.scale(Math.pow(accelNoise, 2)).getDDRM();
		Hy = new DMatrixRMaj(new double[][] { { 1d, 0d } });
		Ry = new SimpleMatrix(new double[][] {{ Math.pow(measurementNoise, 2) }}).getDDRM();
		filterY.configure(Ay, Qy, Hy); 
		P0y = new DMatrixRMaj(new double[][] { { 1, 1 }, { 1, 1 } });
		//y1 = new DMatrixRMaj(new double[][] {{ points.get(fred).y }, { points.get(fred).t  }});
		y1 = new DMatrixRMaj(new double[][] {{ points.get(fred).y }, { 0d  }});//?
	    filterY.setState(y1, P0y);
		
		// Collect data.
		for (int i = 0; i < (SAMPLES + 1) && i < points.size(); i++) {
			fred = points.size() -(SAMPLES + 1)+i;
			fred = (fred > 0)? fred: 0;
			filterX.predict();
			filterY.predict();

			// x = A * y + B * u + pNoise
			//x1 = A.operate((new ArrayRealVector(new double[] {points.get(i).x,points.get(i).t}))).add(B.operate(u)).add(pNoise);
			x1 = new SimpleMatrix(new double[][] {{points.get(fred).x}}).getDDRM();

			// y = A * y + B * u + pNoise
			//y1 = A.operate((new ArrayRealVector(new double[] {points.get(i).y,points.get(i).t}))).add(B.operate(u)).add(pNoise);
			y1 = new SimpleMatrix(new double[][] {{points.get(fred).y}}).getDDRM();

			// z = H * y + m_noise

	
			filterX.update(x1, Rx);
			// z = H * y + m_noise

	
			filterY.update(y1, Ry);

		}

		Point3D ppoint = new Point3D();
		double positionX = filterX.getState().get(0, 0);
		double velocityX = filterX.getState().get(1, 0);
		double positionY = filterY.getState().get(0, 0);
		double velocityY = filterY.getState().get(1, 0);
		System.out.println("velocityX:"+velocityX+" velocityY:"+velocityY);
		ppoint.x = positionX; ppoint.y = positionY; ppoint.t = t;		
		ppoints.add(ppoint);
		Point3D vppoint = new Point3D();
		vppoint.x = velocityX; vppoint.y = velocityY; vppoint.t = t;		
		vppoints.add(vppoint);

	}	

//	private void predictAndDraw(double x, double y, double t) {
//		Point3D point = new Point3D();
//		point.x = x; point.y = y; point.t = t;
//		points.add(point);
//		// Collect data.
//		final WeightedObservedPoints obs = new WeightedObservedPoints();
//		for (int i = 0; i < 6 && i < points.size(); i++) {
//			int fred = points.size() - 6 +i;
//			fred = (fred > 0)? fred: 0;
//			obs.add(points.get(fred).t, points.get(fred).x);
//
//		}
//
//		// Instantiate a third-degree polynomial fitter.
//		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(1);
//
//		// Retrieve fitted parameters (coefficients of the polynomial function).
//		final double[] coeff = fitter.fit(obs.toList());
//		// Collect data.
//		final WeightedObservedPoints obs2 = new WeightedObservedPoints();
//		for (int i = 0; i < 6 && i < points.size(); i++) {
//			int fred = points.size() - 6+i;
//			fred = (fred > 0)? fred: 0;
//			obs2.add(points.get(fred).t, points.get(fred).y);
//
//		}
//
//		// Instantiate a third-degree polynomial fitter.
//		final PolynomialCurveFitter fitter2 = PolynomialCurveFitter.create(1);
//
//		// Retrieve fitted parameters (coefficients of the polynomial function).
//		final double[] coeff2 = fitter2.fit(obs2.toList());
//		Point3D ppoint = new Point3D();
//		ppoint.x = predicted(t, coeff); ppoint.y = predicted(t, coeff2); ppoint.t = t;		
//		ppoints.add(ppoint);
//
//	}
	double predicted(double t, double[] c) {
		double sumOfTerms = 0;
		for (int pow = 0; pow < c.length; pow++) {
			sumOfTerms += Math.pow(t, pow)*c[pow];
		}
		return sumOfTerms;
	}
	double addNoise(double noiseless) {
		return NOISE_VALUE * (LinearFit.rand.nextGaussian())+noiseless;
	}
	class PaintDemo{



		PaintDemo(){
			jfrm.setSize(1000, 800);
			jfrm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			doRender.addActionListener(new ActionListener() {

				@Override
				public void actionPerformed(ActionEvent e) {
					render = true;
					NOISE_VALUE = Double.parseDouble(minx.getText());
					SAMPLES = Integer.parseInt(s.getText());
					try {
						charlie.doASquare();
					} catch (InterruptedException ex) {
						ex.printStackTrace();
					}

					charlie.repaint();				
				}

			});
			jfrm.setLayout(new BorderLayout());
			//			canvas.setSize(1000, 750);
			//			canvas.setVisible(true);
			JPanel temp = new JPanel();
			temp.add(maxxl);		
			temp.add(maxx);
			temp.add(minxl);		
			temp.add(minx);
			temp.add(sl);		
			temp.add(s);
			temp.add(doRender);
			//			canvas.setVisible(true);
			jfrm.add(temp, BorderLayout.NORTH);
			jfrm.add(charlie, BorderLayout.CENTER);

			//jfrm.add(diamond);

			jfrm.setVisible(true);
		}

	}
	public static void main(String args[]) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				charlie = new LinearFit();
				charlie.new PaintDemo();
				charlie.new KalmanX(0.0,0.0);
				charlie.new KalmanY(0.0,0.0);
				try {
					charlie.doASquare();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}

			}
		});

	}

}
