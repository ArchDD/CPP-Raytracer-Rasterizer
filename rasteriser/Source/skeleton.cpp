#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <omp.h>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec2;
using glm::ivec2;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */

//#define FRUSTUM // Uncomment this for naive frustum culling

const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;
float focalLength = 250.0f;

vec3 cameraPos( 0, 0, -2.5f );
mat3 cameraRot = mat3(0.0f);
float yaw = 0; // Yaw angle controlling camera rotation around y-axis

vec3 currentColor;
vec3 currentNormal;
vec3 currentReflectance;

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

vec3 lightPos(0,-0.5,0.5f);
vec3 lightPower = 14.0f*vec3( 1, 1, 1 );
vec3 indirectLightPowerPerArea = 0.5f*vec3( 1, 1, 1 );
bool isUpdated = true;

bool MULTITHREADING_ENABLED = false;
int NUM_THREADS; // Set by code
int SAVED_THREADS; // Stores thread value when changed

bool BACKFACE_CULLING_ENABLED = false;

/* KEY STATES                                                                  */
bool OMP_key_pressed = false;
bool thread_add_key_pressed = false;
bool thread_subtract_key_pressed = false;
bool backface_key_pressed = false;

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
vector<Triangle> triangles;
vector<Triangle> activeTriangles;
Frustum frustum;
float minX = -numeric_limits<float>::max();
float minY = minX;
float minZ = minX;
float maxX = +numeric_limits<float>::max();
float maxY = maxX;
float maxZ = maxX;

void Update();
void Draw();
void VertexShader( const vec3& v, Pixel& p );
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result );
void Breshenham(Pixel a, Pixel b, vector<Pixel>& result);
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color );
void ComputePolygonRows( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels , vec3 color, vec3 normal);
void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels , vec3 color, vec3 normal);
void DrawPolygon( const vector<Vertex>& vertices , vec3 color, vec3 normal);
void PixelShader( const Pixel& p , vec3 color, vec3 normal);
bool InFrustum(vec3 v);
bool InClippingVolume(vec3 v);
bool InCuboid(glm::vec4 v);


int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );

	// Generate the Cornell Box
	LoadTestModel( triangles );

	cameraRot[1][1] = 1.0f;

	// Request as many threads as the system can provide
	NUM_THREADS = omp_get_max_threads();
    omp_set_num_threads(NUM_THREADS);


    // Set NUM_THREADS to how many the system can actually provide
    #pragma omp parallel
    {
    	int ID = omp_get_thread_num();
    	if(ID == 0)
    		NUM_THREADS = omp_get_num_threads();
    	    SAVED_THREADS = NUM_THREADS;
    }

    if(MULTITHREADING_ENABLED)
    {
    	cout << "Multithreading enabled with " << NUM_THREADS << " threads" << endl;
    }
    else
    	omp_set_num_threads(1);

	t = SDL_GetTicks();	// Set start value for timer.

	while( NoQuitMessageSDL() )
	{
			Update();
			if (isUpdated)
			{
				Draw();
				isUpdated = false;
			}
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();

	//cameraPos.x+=0.05f;isUpdated = true;
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	// Clear the screen before drawing the next frame
	for( int y=0; y<SCREEN_HEIGHT; ++y )
	{
		for( int x=0; x<SCREEN_WIDTH; ++x )
		{
			vec3 color( 0.0f, 0.0f, 0.0f );
			depthBuffer[y][x] = 0.0f;
			PutPixelSDL( screen, x, y, color );
		}
	}

	// Adjust camera transform
	vec3 right(cameraRot[0][0], cameraRot[0][1], cameraRot[0][2]);
	vec3 down(cameraRot[1][0], cameraRot[1][1], cameraRot[1][2]);
	vec3 forward(cameraRot[2][0], cameraRot[2][1], cameraRot[2][2]);
	Uint8* keystate = SDL_GetKeyState( 0 );

	if(!OMP_key_pressed && keystate[SDLK_4])
	{
		MULTITHREADING_ENABLED = !MULTITHREADING_ENABLED;
		if(!MULTITHREADING_ENABLED)
			NUM_THREADS = 1;
		else
			NUM_THREADS = SAVED_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Multithreading toggled to " << MULTITHREADING_ENABLED << endl;
		OMP_key_pressed = true;
	}
	else if (!keystate[SDLK_4])
		OMP_key_pressed = false;

	if(!thread_subtract_key_pressed && keystate[SDLK_5])
	{
		NUM_THREADS--;
		SAVED_THREADS = NUM_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Threads decreased to " << NUM_THREADS << endl;
		thread_subtract_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_5])
		thread_subtract_key_pressed = false;

	if(!thread_add_key_pressed && keystate[SDLK_6])
	{
		NUM_THREADS++;
		SAVED_THREADS = NUM_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Threads increased to " << NUM_THREADS << endl;
		thread_add_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_6])
		thread_add_key_pressed = false;

	if(!backface_key_pressed && keystate[SDLK_7])
	{
		if(BACKFACE_CULLING_ENABLED)
		{
			BACKFACE_CULLING_ENABLED = false;
			// Reactive all triangles
			for( size_t i = 0; i < triangles.size(); ++i )
			{
				triangles[i].isCulled = false;
			}
		}
		else
			BACKFACE_CULLING_ENABLED = true;
		cout << "Backface culling toggled " << endl;
		backface_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_7])
		backface_key_pressed = false;

	if( keystate[SDLK_UP] )
	{
		// Move camera forward
		cameraPos += 0.05f*forward;
		isUpdated = true;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 0.05f*forward;
		isUpdated = true;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Rotate camera to the left
		yaw += 0.0025f;
		isUpdated = 1;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Rotate camera to the right
		yaw -= 0.0025f;
		isUpdated = true;
	}

	// Light movement controls
	if (keystate[SDLK_w])
	{
		lightPos.z += 0.05f;
		isUpdated = true;
	}
	else if (keystate[SDLK_s])
	{
		lightPos.z -= 0.05f;
		isUpdated = true;
	}

	if (keystate[SDLK_a])
	{
		lightPos.x -= 0.05f;
		isUpdated = true;;
	}
	else if (keystate[SDLK_d])
	{
		lightPos.x += 0.05f;
		isUpdated = true;
	}

	if (isUpdated)
	{
		// Update camera rotation matrix
		float c = cos(yaw);
		float s = sin(yaw);
		cameraRot[0][0] = c;
		cameraRot[0][2] = s;
		cameraRot[2][0] = -s;
		cameraRot[2][2] = c;

#ifdef FRUSTUM
		// Calculate frustum volume
		vec3 near = cameraPos + 0.1f*forward;
		vec3 far = cameraPos + 10.0f*forward;
		frustum.nearTopLeft = near*(-SCREEN_WIDTH/2.0f)/focalLength;
		frustum.nearTopRight = near*(SCREEN_WIDTH/2.0f)/focalLength;
		frustum.nearBottomLeft = near*(-SCREEN_HEIGHT/2.0f)/focalLength;
		frustum.nearBottomRight = near*(SCREEN_HEIGHT/2.0f)/focalLength;
		frustum.farTopLeft = far*(-SCREEN_WIDTH/2.0f)/focalLength;
		frustum.farTopRight = far*(SCREEN_WIDTH/2.0f)/focalLength;
		frustum.farBottomLeft = far*(-SCREEN_HEIGHT/2.0f)/focalLength;
		frustum.farBottomRight = far*(SCREEN_HEIGHT/2.0f)/focalLength;

		for( size_t i = 0; i < triangles.size(); ++i )
		{
			if (InFrustum(triangles[i].v0) && InFrustum(triangles[i].v1) && InFrustum(triangles[i].v2))
				triangles[i].isCulled = true;
			else
				triangles[i].isCulled = false;
		}
#endif
		// Backface culling
		if (BACKFACE_CULLING_ENABLED)
		{
			for( size_t i = 0; i < triangles.size(); ++i )
			{
				if (dot(forward,triangles[i].normal) > 0.8f)
				{
					triangles[i].isCulled = true;
				}
				else
					triangles[i].isCulled = false;

				// Cull when all vertices are behind
				vec3 v0 = normalize(triangles[i].v0 - cameraPos);
				vec3 v1 = normalize(triangles[i].v1 - cameraPos);
				vec3 v2 = normalize(triangles[i].v2 - cameraPos);

				float d0 = dot(v0, forward);
				float d1 = dot(v1, forward);
				float d2 = dot(v2, forward);
				if (d0 < 0.6f && d1 < 0.6f && d2 < 0.6f)
				{
					triangles[i].isCulled = true; printf("ya");
				}

				/*if (i == 0)
					printf("%f %f %f\n", d0, d1, d2);*/
			}
		}

		// Calculate clipping volume
		vec3 dTopLeft(-SCREEN_WIDTH/2.0f, -SCREEN_HEIGHT/2.0f, focalLength);
		vec3 dTopRight(SCREEN_WIDTH/2.0f, SCREEN_HEIGHT/2.0f, focalLength);
		vec3 dBottomLeft(-SCREEN_WIDTH/2.0f, SCREEN_HEIGHT/2.0f, focalLength);
		vec3 dBottomRight(SCREEN_WIDTH/2.0f, SCREEN_HEIGHT/2.0f, focalLength);
		/*vec3 dTopLeft(200-SCREEN_WIDTH/2.0f, 200-SCREEN_HEIGHT/2.0f, focalLength);
		vec3 dTopRight(300-SCREEN_WIDTH/2.0f, 200-SCREEN_HEIGHT/2.0f, focalLength);
		vec3 dBottomLeft(200-SCREEN_WIDTH/2.0f, 300-SCREEN_HEIGHT/2.0f, focalLength);
		vec3 dBottomRight(300-SCREEN_WIDTH/2.0f, 300-SCREEN_HEIGHT/2.0f, focalLength);*/
		float near = 0.1f/focalLength, far = 100.0/focalLength;
		vec3 nearTopLeft = cameraPos+((cameraRot*dTopLeft)*near);
		vec3 nearTopRight = cameraPos+((cameraRot*dTopRight)*near);
		vec3 nearBottomLeft = cameraPos+((cameraRot*dBottomLeft)*near);
		vec3 nearBottomRight = cameraPos+((cameraRot*dBottomRight)*near);

		vec3 farTopLeft = cameraPos+((cameraRot*dTopLeft)*far);
		vec3 farTopRight = cameraPos+((cameraRot*dTopRight)*far);
		vec3 farBottomLeft = cameraPos+((cameraRot*dBottomLeft)*far);
		vec3 farBottomRight = cameraPos+((cameraRot*dBottomRight)*far);

		// perspective transform
		glm::mat4 pyramid = glm::mat4(0.0f);

		/*pyramid[0][0] = 2.0f/float(SCREEN_WIDTH);
		pyramid[1][1] = 2.0f/float(SCREEN_HEIGHT);
		pyramid[2][2] = 1.0f/float(focalLength);
		pyramid[3][3] = 1.0f;

		glm::mat4 cuboid = glm::mat4(0.0f);
		cuboid[0][0] = 1.0f;
		cuboid[1][1] = 1.0f;
		cuboid[2][2] = -1.0f;
		cuboid[2][3] = -1.0f;*/

		pyramid[0][0] = 2.0f/float(SCREEN_WIDTH);
		pyramid[1][1] = 2.0f/float(SCREEN_HEIGHT);
		pyramid[2][2] = 1.0f/(focalLength);
		//pyramid[3][2] = (-far*near)/(far - near);
		pyramid[3][3] = 1.0f;

		/*pyramid[0][0] = 2.0f/float(SCREEN_WIDTH);
		pyramid[1][1] = 2.0f/float(SCREEN_HEIGHT);
		pyramid[2][2] = 1.0f/(far - near);
		pyramid[3][2] = (-far*near)/(far - near);
		pyramid[2][3] = 1.0f;*/

		glm::vec4 ntl(nearTopLeft.x, nearTopLeft.y, nearTopLeft.z, 1.0f);
		glm::vec4 ntr(nearTopRight.x, nearTopRight.y, nearTopRight.z, 1.0f);
		glm::vec4 nbl(nearBottomLeft.x, nearBottomLeft.y, nearBottomLeft.z, 1.0f);
		glm::vec4 nbr(nearBottomRight.x, nearBottomRight.y, nearBottomRight.z, 1.0f);
		glm::vec4 ftl(farTopLeft.x, farTopLeft.y, farTopLeft.z, 1.0f);
		glm::vec4 ftr(farTopRight.x, farTopRight.y, farTopRight.z, 1.0f);
		glm::vec4 fbl(farBottomLeft.x, farBottomLeft.y, farBottomLeft.z, 1.0f);
		glm::vec4 fbr(farBottomRight.x, farBottomRight.y, farBottomRight.z, 1.0f);
		
		/*glm::vec4 cntl = (ntl*pyramid)*cuboid;
		glm::vec4 cntr = (ntr*pyramid)*cuboid;
		glm::vec4 cnbl = (nbl*pyramid)*cuboid;
		glm::vec4 cnbr = (nbr*pyramid)*cuboid;
		glm::vec4 cftl = (ftl*pyramid)*cuboid;
		glm::vec4 cftr = (ftr*pyramid)*cuboid;
		glm::vec4 cfbl = (fbl*pyramid)*cuboid;
		glm::vec4 cfbr = (fbr*pyramid)*cuboid;*/
		glm::vec4 cntl = (ntl*pyramid);
		glm::vec4 cntr = (ntr*pyramid);
		glm::vec4 cnbl = (nbl*pyramid);
		glm::vec4 cnbr = (nbr*pyramid);
		glm::vec4 cftl = (ftl*pyramid);
		glm::vec4 cftr = (ftr*pyramid);
		glm::vec4 cfbl = (fbl*pyramid);
		glm::vec4 cfbr = (fbr*pyramid);

		//div z
		cntl = cntl/cntl[3];
		cntr = cntr/cntr[3];
		cnbl = cnbl/cnbl[3];
		cnbr = cnbr/cnbr[3];
		cftl = cftl/cftl[3];
		cftr = cftr/cftr[3];
		cfbl = cfbl/cfbl[3];
		cfbr = cfbr/cfbr[3];

		float nMaxX = max(cntl.x, max(cntr.x, max(cnbl.x, cnbr.x)));
		float nMaxY = max(cntl.y, max(cntr.y, max(cnbl.y, cnbr.y)));
		float nMaxZ = max(cntl.z, max(cntr.z, max(cnbl.z, cnbr.z)));

		float fMaxX = max(cftl.x, max(cftr.x, max(cfbl.x, cfbr.x)));
		float fMaxY = max(cftl.y, max(cftr.y, max(cfbl.y, cfbr.y)));
		float fMaxZ = max(cftl.z, max(cftr.z, max(cfbl.z, cfbr.z)));

		float nMinX = min(cntl.x, min(cntr.x, min(cnbl.x, cnbr.x)));
		float nMinY = min(cntl.y, min(cntr.y, min(cnbl.y, cnbr.y)));
		float nMinZ = min(cntl.z, min(cntr.z, min(cnbl.z, cnbr.z)));

		float fMinX = min(cftl.x, min(cftr.x, min(cfbl.x, cfbr.x)));
		float fMinY = min(cftl.y, min(cftr.y, min(cfbl.y, cfbr.y)));
		float fMinZ = min(cftl.z, min(cftr.z, min(cfbl.z, cfbr.z)));

		maxX = max(nMaxX, fMaxX);
		maxY = max(nMaxY, fMaxY);
		maxZ = max(nMaxZ, fMaxZ);

		minX = min(nMinX, fMinX);
		minY = min(nMinY, fMinY);
		minZ = min(nMinZ, fMinZ);

		for( size_t i = 0; i < triangles.size(); ++i )
		{
			vec3 v0 = triangles[i].v0;
			vec3 v1 = triangles[i].v1;
			vec3 v2 = triangles[i].v2;

			/*glm::vec4 hv0(v0.x, v0.y, v0.z, 1.0f);
			glm::vec4 pv0 = (hv0*pyramid)*cuboid;
			glm::vec4 hv1(v1.x, v1.y, v1.z, 1.0f);
			glm::vec4 pv1 = (hv1*pyramid)*cuboid;
			glm::vec4 hv2(v2.x, v2.y, v2.z, 1.0f);
			glm::vec4 pv2 = (hv2*pyramid)*cuboid;*/

			glm::vec4 hv0(v0.x, v0.y, v0.z, 1.0f);
			glm::vec4 pv0 = hv0*pyramid;
			glm::vec4 hv1(v1.x, v1.y, v1.z, 1.0f);
			glm::vec4 pv1 = hv1*pyramid;
			glm::vec4 hv2(v2.x, v2.y, v2.z, 1.0f);
			glm::vec4 pv2 = hv2*pyramid;

			//divide by z
			pv0 = pv0/pv0[3];
			pv1 = pv1/pv1[3];
			pv2 = pv2/pv2[3];

		

			if (InCuboid(pv0) && InCuboid(pv1) && InCuboid(pv2))
			{
				triangles[i].isCulled = false;
			}
			else
			{
				triangles[i].isCulled = true;
				//printf("ay\n");exit(0);
			}

			if (i == 2)
			{
				printf("minX %f maxX %f minY %f maxY %f minZ %f maxZ %f\n", minX, maxX, minY, maxY, minZ, maxZ);
				printf("pv0 %f %f %f\n", pv0.x, pv0.y, pv0.z);
				printf("pv1 %f %f %f\n", pv1.x, pv1.y, pv1.z);
				printf("pv2 %f %f %f\n", pv2.x, pv2.y, pv2.z);

				printf("pv0 %f %f %f\n", pv0.x, pv0.y, pv0.z);
				printf("pv1 %f %f %f\n", pv1.x, pv1.y, pv1.z);
				printf("pv2 %f %f %f\n", pv2.x, pv2.y, pv2.z);
				glm::vec4 norm = glm::vec4(cameraPos.x, cameraPos.y, cameraPos.z, 1.0f)*pyramid;
				norm = norm/norm[3];
				printf("cam x %f y %f z %f\n", cameraPos.x, cameraPos.y, cameraPos.z);
				printf("norm cam x %f y %f z %f\n", cameraPos.x, cameraPos.y, cameraPos.z);
				printf("ntl x %f y %f z %f\n", nearTopLeft.x, nearTopLeft.y, nearTopLeft.z);
				printf("isCulled: %d\n", triangles[i].isCulled);
				//exit(0);
			}
		}
	}
}

bool InFrustum(vec3 v)
{
	if (v.x < frustum.nearTopRight.x && v.x < frustum.farTopRight.x &&
		v.x > frustum.nearTopLeft.x && v.x > frustum.farTopLeft.x &&
		v.y < frustum.nearBottomLeft.y && v.y < frustum.farBottomLeft.y &&
		v.y < frustum.nearTopLeft.y && v.y < frustum.farTopLeft.y &&
		v.z < frustum.farTopLeft.z && v.z > frustum.nearTopLeft.z
		)
	{
		return true;
	}

	return false;
}

bool InCuboid(glm::vec4 v)
{
	if (v.x > minX && v.x < maxX && v.y > minY && v.y < maxY && v.z > minZ && v.z < maxZ)
	{
		return true;
	}

	return false;
}

bool InClippingVolume(vec3 v)
{
	if (v.x >= minX && v.x <= maxX && v.y >= minY && v.y <= maxY && v.z >= minZ && v.z <= maxZ)
	{
		return true;
	}

	return false;
}

void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	currentReflectance = vec3(1.0f,1.0f,1.0f);

	#pragma omp parallel for schedule(dynamic)
	for( size_t i = 0; i < triangles.size(); ++i )
	{
		//if (i != 2) triangles[i].isCulled = true;
		if (!triangles[i].isCulled)
		{
			printf("%d , %d - ",triangles.size(), i);
			// Get the 3 vertices of the triangle
			vector<Vertex> vertices(3);
			vertices[0].position = triangles[i].v0;
			vertices[1].position = triangles[i].v1;
			vertices[2].position = triangles[i].v2;
			DrawPolygon( vertices , triangles[i].color, triangles[i].normal);
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}

// Compute 2D screen position from 3D position
void VertexShader( const Vertex& v, Pixel& p )
{
	// We need to adjust the vertex positions based on the camera position and rotation
	vec3 pos = (v.position - cameraPos) * cameraRot;

	// prevent dividing by 0
	/*if(pos.z < 0.01f)
		pos.z = 0.01f;*/
	if (pos.z > 0.0f)
	{
		// Store the 3D position of the vertex in the pixel. Divide by the Z value so the value interpolates correctly over the perspective
		p.pos3d = pos / pos.z;

		// Calculate depth of pixel, inversed so the value interpolates correctly over the perspective
		p.zinv = 1.0f / pos.z;

		// Calculate 2D screen position and place (0,0) at top left of the screen
		p.x = int(focalLength * (pos.x * p.zinv)) + (SCREEN_WIDTH / 2.0f);
		p.y = int(focalLength * (pos.y * p.zinv)) + (SCREEN_HEIGHT / 2.0f);

		/*if (p.x > SCREEN_WIDTH)
			p.x = SCREEN_WIDTH;
		else if (p.x < 0)
			p.x = 0;
		if (p.y > SCREEN_HEIGHT)
			p.y = SCREEN_HEIGHT;
		else if (p.y < 0)
			p.y = 0;*/
	}
}

// Calculate per pixel lighting
void PixelShader( const Pixel& p , vec3 color, vec3 normal)
{
	int x = p.x;
	int y = p.y;

	// Multiply pixel 3d position by the z value to get the origin position from the inverse
	vec3 pPos3d(p.pos3d);
	pPos3d *= p.pos3d.z;

	// Calculate lighting
	float r = glm::distance(pPos3d, lightPos);
	float A = 4*M_PI*(r*r);
	
	vec3 rDir = glm::normalize(lightPos - pPos3d);
	vec3 nDir = normal;

	vec3 D = (lightPower * max(glm::dot(rDir,nDir), 0.0f)) / A;

	vec3 pixelColor = currentReflectance * (D + indirectLightPowerPerArea) * color;

	PutPixelSDL( screen, x, y, pixelColor );
}

// Draws a line between two points
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color, vec3 normal)
{
	Pixel delta = a - b;
	PixelAbs(delta);
	
	// DDA
	/*int pixels = max( delta.x, delta.y ) + 1;

	vector<Pixel> line( pixels );

	Interpolate( a, b, line );*/

	// Breshenam
	int pixels = b.x-a.x;
	vector<Pixel> line (pixels);

	Breshenham(a,b,line);

	for(int i = 0; i < pixels; ++i)
	{
		// Ensure pixel is on the screen and is closer to the camera than the current value in the depth buffer
		if(line[i].y < SCREEN_HEIGHT && line[i].y >= 0 && line[i].x < SCREEN_WIDTH && line[i].x >= 0 && line[i].zinv > depthBuffer[line[i].y][line[i].x])
		{
			depthBuffer[line[i].y][line[i].x] = line[i].zinv;
			PixelShader(line[i], color, normal);
		}
	}
}

// Interpolates between two Pixels
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result )
{
	int N = result.size();
	Pixel delta = b-a;

	fPixel step(delta);

	step = (step / float(max(N-1,1)));

	fPixel current( a );

	for( int i=0; i<N; ++i )
	{
		result[i].x = current.x;
		result[i].y = current.y;
		result[i].zinv = current.zinv;
		result[i].pos3d = current.pos3d;
		current.x += step.x;
		current.y += step.y;
		current.zinv += step.zinv;
		current.pos3d += step.pos3d;
	}
}

void Breshenham(Pixel a, Pixel b, vector<Pixel>& result)
{
	int x = a.x;
	int y = a.y;
	int dx = b.x-a.x, dy = b.y-a.y;
	int dx2 = 2*dx, dy2 = 2*dy;
	int dydx2 = dy2 - dx2;
	int d = dy2 - dx;

	float zinv = (b.zinv - a.zinv)/float(dx);
	vec3 pos3d = (b.pos3d - a.pos3d)/float(dx);

	for (int i = 0;  i < dx; i++)
	{
		x+=1;
		if (d<0)
		{
			d+=dy2;
		}
		else
		{
			y+=1;
			d+=dydx2;
		}
		result[i].x = x;
		result[i].y = y;
		result[i].zinv = a.zinv+zinv*float(i);
		result[i].pos3d = a.pos3d+pos3d*float(i);
	}
}

void ComputePolygonRows( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels )
{
	// 1. Find max and min y-value of the polygon
	// and compute the number of rows it occupies.

	int maxY = max(max(vertexPixels[0].y, vertexPixels[1].y), vertexPixels[2].y);
	int minY = min(min(vertexPixels[0].y, vertexPixels[1].y), vertexPixels[2].y);

	int ROWS = maxY - minY + 1;

	// 2. Resize leftPixels and rightPixels
	// so that they have an element for each row.

	leftPixels.resize( ROWS );
	rightPixels.resize( ROWS );

	// 3. Initialize the x-coordinates in leftPixels
	// to some really large value and the x-coordinates
	// in rightPixels to some really small value.

	for( int i = 0; i < ROWS; ++i )
	{
		leftPixels[i].x = +numeric_limits<int>::max();
		rightPixels[i].x = -numeric_limits<int>::max();
	}

	// 4. Loop through all edges of the polygon and use
	// linear interpolation to find the x-coordinate for
	// each row it occupies. Update the corresponding
	// values in rightPixels and leftPixels.

	for(int i = 0; i < 3; i++)
	{
		int j = (i + 1) % 3; // Ensure all 3 edges are looped through
		// Adjust vertex positions to have y value 0 at minY so Y coordinates map to array indicies
		Pixel v1 (vertexPixels[i].x, vertexPixels[i].y - minY, vertexPixels[i].zinv, vertexPixels[i].pos3d);
		Pixel v2 (vertexPixels[j].x, vertexPixels[j].y - minY, vertexPixels[j].zinv, vertexPixels[j].pos3d);

		int edgePixels = abs(vertexPixels[i].y - vertexPixels[j].y) + 1; // Calculate number of rows this edge occupies
		vector<Pixel> edgeResult(edgePixels); // Create array of ivec2 with number of rows
		Interpolate(v1,v2,edgeResult); // Interpolate between the two vertices

		for(int k = 0; k < edgePixels; k++)
		{
			if(edgeResult[k].x < leftPixels[edgeResult[k].y].x)
			{
				leftPixels[edgeResult[k].y].x = edgeResult[k].x;
				leftPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				leftPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				leftPixels[edgeResult[k].y].pos3d = edgeResult[k].pos3d;
			}

			if(edgeResult[k].x > rightPixels[edgeResult[k].y].x)
			{
				rightPixels[edgeResult[k].y].x = edgeResult[k].x;
				rightPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				rightPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				rightPixels[edgeResult[k].y].pos3d = edgeResult[k].pos3d;
			}
		}
	}
}

// Draw a line for each row of the triangle
void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels , vec3 color, vec3 normal)
{
	for(int i = 0; i < leftPixels.size(); i++)
	{
		DrawLineSDL(screen, leftPixels[i],rightPixels[i],color, normal);
	}
}

void DrawPolygon( const vector<Vertex>& vertices , vec3 color, vec3 normal)
{
	int V = vertices.size();
	vector<Pixel> vertexPixels( V );

	for( int i=0; i<V; ++i )
		VertexShader( vertices[i], vertexPixels[i] );

	vector<Pixel> leftPixels;
	vector<Pixel> rightPixels;

	ComputePolygonRows( vertexPixels, leftPixels, rightPixels );
	DrawRows( leftPixels, rightPixels , color, normal);
}
