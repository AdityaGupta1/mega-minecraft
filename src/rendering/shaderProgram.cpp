#include "shaderProgram.hpp"

#include "util/utils.hpp"
#include <iostream>
#include "renderingUtils.hpp"
#include "structs.hpp"
#include <regex>

ShaderProgram::ShaderProgram()
    : vertShader(-1), fragShader(-1), compShader(-1), prog(-1), 
      attr_pos(-1), attr_nor(-1), attr_uv(-1), 
      unif_modelMat(-1), unif_viewProjMat(-1), unif_invViewProjMat(-1), unif_viewMat(-1), unif_invViewMat(-1), unif_projMat(-1), unif_sunViewProjMat(-1),
      unif_sunDir(-1), unif_moonDir(-1), unif_fogColor(-1),
      tex_blockDiffuse(-1), tex_bufColor(-1), tex_shadowMap(-1), tex_volume(-1)
{}

void printShaderInfoLog(int shader)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 0)
    {
        infoLog = new GLchar[infoLogLen];
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        std::cerr << "ShaderInfoLog:" << "\n" << infoLog << "\n";
        delete[] infoLog;
    }
}

void printLinkInfoLog(int prog)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 0)
    {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
        std::cerr << "LinkInfoLog:" << "\n" << infoLog << "\n";
        delete[] infoLog;
    }
}

void ShaderProgram::createUniformVariables()
{
    attr_pos = glGetAttribLocation(prog, "vs_pos");
    attr_nor = glGetAttribLocation(prog, "vs_nor");
    attr_uv = glGetAttribLocation(prog, "vs_uv");

    unif_modelMat = glGetUniformLocation(prog, "u_modelMat");
    unif_viewProjMat = glGetUniformLocation(prog, "u_viewProjMat");
    unif_invViewProjMat = glGetUniformLocation(prog, "u_invViewProjMat");
    unif_viewMat = glGetUniformLocation(prog, "u_viewMat");
    unif_invViewMat = glGetUniformLocation(prog, "u_invViewMat");
    unif_projMat = glGetUniformLocation(prog, "u_projMat");
    unif_sunViewProjMat = glGetUniformLocation(prog, "u_sunViewProjMat");

    unif_sunDir = glGetUniformLocation(prog, "u_sunDir");
    unif_moonDir = glGetUniformLocation(prog, "u_moonDir");
    unif_fogColor = glGetUniformLocation(prog, "u_fogColor");

    tex_blockDiffuse = glGetUniformLocation(prog, "tex_blockDiffuse");
    tex_bufColor = glGetUniformLocation(prog, "tex_bufColor");
    tex_shadowMap = glGetUniformLocation(prog, "tex_shadowMap");

    if ((tex_volume = glGetUniformLocation(prog, "img_volume")) == -1) {
        tex_volume = glGetUniformLocation(prog, "tex_volume");
    }
}

bool checkShaderCompiled(GLint shader)
{
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        printShaderInfoLog(shader);
        return false;
    }

    return true;
}

bool checkProgLinked(GLint prog)
{
    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        printLinkInfoLog(prog);
        return false;
    }

    return true;
}

static const std::string definesText = Utils::readFile("shaders/defines.glsl");

std::string readFileAndInsertDefines(const std::string& file)
{
    return std::regex_replace(Utils::readFile(file), std::regex("#include defines\\.glsl"), definesText);
}

bool ShaderProgram::create(const std::string& vertFile, const std::string& fragFile)
{
    std::cout << "creating shader from " << vertFile << " and " << fragFile << "...    ";

    vertShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    prog = glCreateProgram();

    const std::string vertText = readFileAndInsertDefines(vertFile);
    const std::string fragText = readFileAndInsertDefines(fragFile);

    const char* vertChars = vertText.c_str();
    const char* fragChars = fragText.c_str();

    glShaderSource(vertShader, 1, (const char**)&vertChars, 0);
    glShaderSource(fragShader, 1, (const char**)&fragChars, 0);
    glCompileShader(vertShader);
    glCompileShader(fragShader);

    if (!checkShaderCompiled(vertShader) || !checkShaderCompiled(fragShader))
    {
        return false;
    }

    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);
    glLinkProgram(prog);

    if (!checkProgLinked(prog))
    {
        return false;
    }

    attr_pos = glGetAttribLocation(prog, "vs_pos");
    attr_nor = glGetAttribLocation(prog, "vs_nor");
    attr_uv = glGetAttribLocation(prog, "vs_uv");

    createUniformVariables();

    std::cout << "done" << std::endl;
    return true;
}

bool ShaderProgram::createCompute(const std::string& compFile)
{
    std::cout << "creating compute shader from " << compFile << "...    ";

    compShader = glCreateShader(GL_COMPUTE_SHADER);
    prog = glCreateProgram();

    const std::string compText = readFileAndInsertDefines(compFile);

    const char* compChars = compText.c_str();

    glShaderSource(compShader, 1, (const char**)&compText, 0);
    glCompileShader(compShader);

    if (!checkShaderCompiled(compShader))
    {
        return false;
    }

    glAttachShader(prog, compShader);
    glLinkProgram(prog);

    if (!checkProgLinked(prog))
    {
        return false;
    }

    createUniformVariables();

    std::cout << "done" << std::endl;
    return true;
}

void ShaderProgram::useMe() const
{
    glUseProgram(prog);
}

void ShaderProgram::setModelMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_modelMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setViewProjMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_viewProjMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setInvViewProjMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_invViewProjMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setViewMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_viewMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setInvViewMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_invViewMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setProjMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_projMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setSunViewProjMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_sunViewProjMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setSunDir(const glm::vec4& dir) const
{
    useMe();
    glUniform4fv(unif_sunDir, 1, &dir[0]);
}

void ShaderProgram::setMoonDir(const glm::vec4& dir) const
{
    useMe();
    glUniform4fv(unif_moonDir, 1, &dir[0]);
}

void ShaderProgram::setFogColor(const glm::vec3& col) const
{
    useMe();
    glUniform3fv(unif_fogColor, 1, &col[0]);
}

void ShaderProgram::setTexBlockDiffuse(int tex) const
{
    useMe();
    glUniform1i(tex_blockDiffuse, tex);
}

void ShaderProgram::setTexBufColor(int tex) const
{
    useMe();
    glUniform1i(tex_bufColor, tex);
}

void ShaderProgram::setTexShadowMap(int tex) const
{
    useMe();
    glUniform1i(tex_shadowMap, tex);
}

void ShaderProgram::setTexVolume(int tex) const
{
    useMe();
    glUniform1i(tex_volume, tex);
}

void ShaderProgram::draw(Drawable& d) const
{
    useMe();

    if (d.getIdxCount() < 0)
    {
        throw std::out_of_range("drawable has no elements");
    }

    if (d.bindVerts())
    {
        if (attr_pos != -1)
        {
            glEnableVertexAttribArray(attr_pos);
            glVertexAttribPointer(attr_pos, 3, GL_FLOAT, false, sizeof(Vertex), (void*)0);
        }

        if (attr_nor != -1)
        {
            glEnableVertexAttribArray(attr_nor);
            glVertexAttribPointer(attr_nor, 3, GL_FLOAT, false, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
        }

        if (attr_uv != -1)
        {
            glEnableVertexAttribArray(attr_uv);
            glVertexAttribPointer(attr_uv, 2, GL_FLOAT, false, sizeof(Vertex), (void*)(2 * sizeof(glm::vec3)));
        }
    }
    else if (d.bindFullscreenTriInfo())
    {
        if (attr_pos != -1)
        {
            glEnableVertexAttribArray(attr_pos);
            glVertexAttribPointer(attr_pos, 3, GL_FLOAT, false, 5 * sizeof(float), (void*)0);
        }

        if (attr_uv != -1)
        {
            glEnableVertexAttribArray(attr_uv);
            glVertexAttribPointer(attr_uv, 2, GL_FLOAT, false, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        }
    }

    d.bindIdx();
    glDrawElements(d.drawMode(), d.getIdxCount(), GL_UNSIGNED_INT, 0);

    RenderingUtils::printGLErrors();
}

void ShaderProgram::dispatchCompute(int groupsX, int groupsY, int groupsZ)
{
    useMe();
    glDispatchCompute(groupsX, groupsY, groupsZ);
}