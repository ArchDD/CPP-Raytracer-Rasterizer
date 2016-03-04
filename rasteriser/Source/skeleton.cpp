#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec2;
using glm::ivec2;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */

const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;
float focalLength = 250.0f;

vec3 cameraPos( 0, 0, -2.0f );
mat3 cameraRot = mat3(0.0f);
float yaw = 0; // Yaw angle controlling camera rotation around y-axis

vec3 currentColor;

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

vec3 lightPos(0,-0.5,-0.7);
vec3 lightPower = 14.0f*vec3( 1, 1, 1 );
vec3 indirectLightPowerPerArea = 0.2f*vec3( 1, 1, 1 );

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
vector<Triangle> triangles;

void Update();
void Draw();
void VertexShader( const vec3& v, Pixel& p );
void Interpolate( ivec2 a, ivec2 b, vector<ivec2>& result );
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result );
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color );
void DrawPolygonEdges( const vector<vec3>& vertices );
void ComputePolygonRows( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels );
void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels );
void DrawPolygon( const vector<Vertex>& vertices );
void PixelShader( const Pixel& p );

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	// Generate the Cornell Box
	LoadTestModel( triangles );

	cameraRot[1][1] = 1.0f;

	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	// Clear the screen before drawing the next frame
	for( int y=0; y<SCREEN_HEIGHT; ++y )
	{
		for( int x=0; x<SCREEN_WIDTH; ++x )
		{
			vec3 color( 0.0, 0.0, 0.0 );
			depthBuffer[y][x] = 0;
			PutPixelSDL( screen, x, y, color );
		}
	}

	// Adjust camera transform
	vec3 right(cameraRot[0][0], cameraRot[0][1], cameraRot[0][2]);
	vec3 down(cameraRot[1][0], cameraRot[1][1], cameraRot[1][2]);
	vec3 forward(cameraRot[2][0], cameraRot[2][1], cameraRot[2][2]);
	Uint8* keystate = SDL_GetKeyState( 0 );

	if( keystate[SDLK_UP] )
	{
		// Move camera forward
		cameraPos += 0.1f*forward;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 0.1f*forward;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Rotate camera to the left
		yaw += 0.1f;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Rotate camera to the right
		yaw -= 0.1f;
	}

	// Update camera rotation matrix
	float c = cos(yaw);
	float s = sin(yaw);
	cameraRot[0][0] = c;
	cameraRot[0][2] = s;
	cameraRot[2][0] = -s;
	cameraRot[2][2] = c;
}

void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	for( size_t i = 0; i < triangles.size(); ++i )
	{
		// Get the 3 vertices of the triangle
		vector<Vertex> vertices(3);
		vertices[0].position = triangles[i].v0;
		vertices[0].normal = triangles[i].normal;
		vertices[0].reflectance = vec2(0.5f,0.5f);

		vertices[1].position = triangles[i].v1;
		vertices[1].normal = triangles[i].normal;
		vertices[1].reflectance = vec2(0.5f,0.5f);

		vertices[2].position = triangles[i].v2;
		vertices[2].normal = triangles[i].normal;
		vertices[2].reflectance = vec2(0.5f,0.5f);

		currentColor = triangles[i].color;

		DrawPolygon( vertices );
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}

// Compute 2D screen position from 3D position
void VertexShader( const Vertex& v, Pixel& p )
{
	// We need to adjust the vertex positions based on the camera position and rotation
	Vertex v2(v);
	v2.position = (v.position - cameraPos) * cameraRot;

	// Calculate 2D screen position and place (0,0) at top left of the screen
	p.zinv = 1.0f / v2.position.z;
	p.x = int(focalLength * (v2.position.x * p.zinv)) + (SCREEN_WIDTH / 2.0f);
	p.y = int(focalLength * (v2.position.y * p.zinv)) + (SCREEN_HEIGHT / 2.0f);

	float r = glm::distance(v.position, lightPos);
	float A = 4*M_PI*(r*r);
	
	vec3 rDir = glm::normalize(v.position - lightPos);
	vec3 nDir = v.normal;

	vec3 D = (lightPower * max(glm::dot(rDir,nDir), 0.0f)) / A;

	//vec3 R = v2.reflectance * (D + indirectLightPowerPerArea);

	p.illumination = (D + indirectLightPowerPerArea) * currentColor;

	//vec3 R = D + indirectLightPowerPerArea;

	//cout << "p illum " << p.illumination.x + p.illumination.y + p.illumination.z << endl;

}

void PixelShader( const Pixel& p )
{
	int x = p.x;
	int y = p.y;

	//cout << p.illumination.z << endl;
	PutPixelSDL( screen, x, y, p.illumination );
}

// Draws a line between two points
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color )
{

	Pixel delta = a - b;
	
	//cout << "a illum is " << a.illumination.x + a.illumination.y + a.illumination.z << endl;
	//cout << "b illum is " << b.illumination.x + b.illumination.y + b.illumination.z << endl;

	//cout << "delta illum is " << delta.illumination.x + delta.illumination.y + delta.illumination.z << endl;

	PixelAbs(delta);

		//cout << "delta abs illum is " << delta.illumination.x + delta.illumination.y + delta.illumination.z << endl;
	
	int pixels = max( delta.x, delta.y ) + 1;

	vector<Pixel> line( pixels );

	Interpolate( a, b, line );

	for(int i = 0; i < pixels; ++i)
	{
		if(line[i].y < SCREEN_HEIGHT && line[i].y >= 0 && line[i].x < SCREEN_WIDTH && line[i].x >= 0 && line[i].zinv > depthBuffer[line[i].y][line[i].x])
		{
			depthBuffer[line[i].y][line[i].x] = line[i].zinv;
			PixelShader(line[i]);
		}
	}
}

// Interpolates between two 2D vectors
void Interpolate( ivec2 a, ivec2 b, vector<ivec2>& result )
{
	int N = result.size();

	vec2 step = vec2(b-a);
	step = step / float(max(N-1,1));
	
	vec2 current( a );

	for( int i=0; i<N; ++i )
	{
		result[i] = current;
		current += step;
	}
}

// Interpolates between two 2D vectors
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result )
{
	int N = result.size();
	Pixel delta = Pixel(b-a);

	//cout << "delta illum is " << delta.illumination.x + delta.illumination.y + delta.illumination.z << endl;
	fPixel step(delta);

	//cout << "step illum is " << step.illumination.x + step.illumination.y + step.illumination.z << endl;

	//cout << " N is " << N << endl;
	step = (step / float(max(N-1,1)));

	//cout << "new step illum is " << step.illumination.x + step.illumination.y + step.illumination.z << endl;

	fPixel current( a );
	//cout << "a illum is " << a.illumination.x + a.illumination.y + a.illumination.z << endl;
	//cout << "b illum is " << b.illumination.x + b.illumination.y + b.illumination.z << endl;
		//cout << "current illum is " << current.illumination.x + current.illumination.y + current.illumination.z << endl;

	for( int i=0; i<N; ++i )
	{
		result[i].x = current.x;
		result[i].y = current.y;
		result[i].zinv = current.zinv;
		result[i].illumination = current.illumination;
		current.x += step.x;
		current.y += step.y;
		current.zinv += step.zinv;
		current.illumination += step.illumination;
		//cout << " illum increased to " << current.illumination.x + current.illumination.y + current.illumination.z << endl;
	}
}

/* Draw every polygon edge
void DrawPolygonEdges( const vector<vec3>& vertices )
{
	int V = vertices.size();
	// Transform each vertex from 3D world position to 2D image position:
	vector<Pixel> projectedVertices( V );
	for( int i=0; i<V; ++i )
	{
		VertexShader( vertices[i], projectedVertices[i] );
	}
	// Loop over all vertices and draw the edge from it to the next vertex:
	for( int i=0; i<V; ++i )
	{
		int j = (i+1)%V; // The next vertex
		vec3 color( 1, 1, 1 );
		DrawLineSDL( screen, projectedVertices[i], projectedVertices[j], color );
	}
}*/

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
		Pixel v1 (vertexPixels[i].x, vertexPixels[i].y - minY, vertexPixels[i].zinv, vertexPixels[i].illumination);
		Pixel v2 (vertexPixels[j].x, vertexPixels[j].y - minY, vertexPixels[j].zinv, vertexPixels[i].illumination);

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
				leftPixels[edgeResult[k].y].illumination = edgeResult[k].illumination;
				//cout << "left illum is " << leftPixels[edgeResult[k].y].illumination.x + leftPixels[edgeResult[k].y].illumination.y + leftPixels[edgeResult[k].y].illumination.z << endl;
			}

			if(edgeResult[k].x > rightPixels[edgeResult[k].y].x)
			{
				rightPixels[edgeResult[k].y].x = edgeResult[k].x;
				rightPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				rightPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				rightPixels[edgeResult[k].y].illumination = edgeResult[k].illumination;
				//cout << "right illum is " << leftPixels[edgeResult[k].y].illumination.x + leftPixels[edgeResult[k].y].illumination.y + leftPixels[edgeResult[k].y].illumination.z << endl;
			}
		}
	}
}

void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels )
{
	for(int i = 0; i < leftPixels.size(); i++)
	{
		DrawLineSDL(screen, leftPixels[i],rightPixels[i],currentColor);
	}
}

void DrawPolygon( const vector<Vertex>& vertices )
{
	int V = vertices.size();
	vector<Pixel> vertexPixels( V );

	for( int i=0; i<V; ++i )
		VertexShader( vertices[i], vertexPixels[i] );

	vector<Pixel> leftPixels;
	vector<Pixel> rightPixels;

	ComputePolygonRows( vertexPixels, leftPixels, rightPixels );

	DrawRows( leftPixels, rightPixels );
}
