#include<iostream>
#include<fstream>
#include<stdio.h>
#include<vector>
#include<string>
#include <sstream>
#include <glm/glm.hpp>
#include <algorithm>

using namespace std;

// Loads a 3D model stored in STL format as a vector of triangles

class LoadSTL
{
public:
	void LoadSTLFile(std::vector<Triangle>& triangles )
	{
		float scale = 0.05f;
		string line;
		// STL doesn't support baked colour information so we need to set the colour manually
		glm::vec3 colour(  0.5f, 0.5f, 0.5f );

		// Start file stream
		ifstream inputStream;
		inputStream.open("Source/enemy1.stl");

		// Clear and reserve triangles
		triangles.clear();
		triangles.reserve( 10000 );

		while (getline(inputStream, line))
		{
			// Search for "outer" substring then extract the 3 vertices
			if(line.find("outer") != string::npos)
			{
				glm::vec3 v1,v2,v3;

				// Get X, Y, Z values of vertex
				for(int i = 0; i < 3; i++)
				{
					string vertexLine; 
					getline(inputStream, vertexLine);
					vector<string> vertexValues = split(vertexLine, ' ');

					float x, y, z;
					x = (float)atof(vertexValues[0].c_str());
					y = (float)atof(vertexValues[1].c_str());
					z = (float)atof(vertexValues[2].c_str());

					if(i == 0)
						v1 = glm::vec3(x,y,z);
					else if(i == 1)
						v2 = glm::vec3(x,y,z);
					else 
						v3 = glm::vec3(x,y,z);
				}
				triangles.push_back( Triangle( v1, v2, v3, colour ) );
			}
			
		}

		// Set scale, compute normal
		for( size_t i=0; i<triangles.size(); ++i )
		{

			triangles[i].v0.x *= -scale;
			triangles[i].v1.x *= -scale;
			triangles[i].v2.x *= -scale;

			triangles[i].v0.z *= -scale;
			triangles[i].v1.z *= -scale;
			triangles[i].v2.z *= -scale;

			triangles[i].v0.y *= -scale;
			triangles[i].v1.y *= -scale;
			triangles[i].v2.y *= -scale;

			triangles[i].ComputeNormal();
		}
	}

	// Takes a string and splits it by a delimiter
	vector<string> split(string str, char delimiter) 
	{
	  vector<string> internal;
	  stringstream ss(str);
	  string tok;
	  
	  while(getline(ss, tok, delimiter)) 
	  {
	    if(tok.size() > 0 && tok != "vertex")
	    	internal.push_back(tok);
	  }
	  
	  return internal;
	}
};