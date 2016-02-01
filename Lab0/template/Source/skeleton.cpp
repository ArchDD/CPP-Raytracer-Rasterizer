#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"

using namespace std;
using glm::vec3;
using glm::mat3;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */

vector<glm::vec3> stars( 1000 ); // Defining stars
const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
//const int RAND_MAX = 2;
SDL_Surface* screen;
int t;

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void Update();
void Draw();
void Interpolate(float a, float b, vector<float>& result );
void Interpolate3( vec3 a, vec3 b, vector<vec3>& result );

int main( int argc, char* argv[] )
{
	uint i;
	for (i = 0; i < stars.size(); i++)
	{
		stars[i].x = (float(rand()) / float(RAND_MAX))*2 - 1.0f;
		stars[i].y = (float(rand()) / float(RAND_MAX))*2 - 1.0f;
		stars[i].z = float(rand()) / float(RAND_MAX);
		printf("x %f, y %f, z %f\n", stars[i].x, stars[i].y, stars[i].z);
	}


	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

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
}

void Draw()
{
	//1D
	/*vector<float> result( SCREEN_WIDTH ); // Create a vector width 10 floats
	Interpolate( 0.0, 1.0, result ); // Fill it with interpolated values
	for( int i=0; i<result.size(); ++i )
		cout << result[i] << " "; // Print the result to the terminal*/

	//3D
	/*
	vector<vec3> result( SCREEN_WIDTH );
	vec3 a(0.0,0.0,0.0);
	vec3 b(1.0,1.0,1.0);
	Interpolate3( a, b, result );
	for( int i=0; i<result.size(); ++i )
	{
		cout << "( "
		<< result[i].x << ", "
		<< result[i].y << ", "
		<< result[i].z << " ) ";
	}

	//Bilinear Interpolation of Colors
	vec3 topLeft(1,0,0);     // red
	vec3 topRight(0,0,1);    // blue
	vec3 bottomRight(0,1,0); // green
	vec3 bottomLeft(1,1,0);  // yellow

	vector<vec3> result ( SCREEN_WIDTH );
	vector<vec3> leftSide( SCREEN_HEIGHT );
	vector<vec3> rightSide( SCREEN_HEIGHT );
	Interpolate3( topLeft, bottomLeft, leftSide );
	Interpolate3( topRight, bottomRight, rightSide );

	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	for( int y=0; y<SCREEN_HEIGHT; ++y )
	{
		for( int x=0; x<SCREEN_WIDTH; ++x )
		{
			//vec3 color( 0.0, 5.0, 0.0 );
			//vec3 color( 0.0, result[x], 0.0 ); // 1D lerp
			//vec3 color( result[SCREEN_WIDTH-x].x, result[x].y, result[x].z); // 3D lerp

			Interpolate3( rightSide[x], leftSide[x], result ); // Bilinear
			vec3 color( result[y].x, result[y].y, result[y].z );
			PutPixelSDL( screen, x, y, color );
		}
	}*/

	// stars
	SDL_FillRect( screen, 0, 0 );

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	uint i = 0;
	for (i = 0; i <stars.size(); i++)
	{
		int xVal = int(stars[i].x * SCREEN_WIDTH);
		int yVal = int(stars[i].y * SCREEN_HEIGHT);
		PutPixelSDL( screen, xVal, yVal, vec3(1,1,1) );
		//printf("stars x %f stars y %f x %d y %d\n", stars[i].x, stars[i].y, xVal, yVal);
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}


void Interpolate(float a, float b, vector<float>& result )
{
	uint i;
	for (i = 0; i < result.size(); i++)
	{
		result[i] = a + i * ( (b-a) / ( result.size()-1 ) );
	}
}

void Interpolate3( vec3 a, vec3 b, vector<vec3>& result )
{
	uint i;
	for (i = 0; i < result.size(); i++)
	{
		result[i].x = a.x + i * ( (b.x-a.x) / ( result.size()-1 ) );
		result[i].y = a.y + i * ( (b.y-a.y) / ( result.size()-1 ) );
		result[i].z = a.z + i * ( (b.z-a.z) / ( result.size()-1 ) );
	}
}
