#pragma once

#include <windows.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <directxcolors.h>
#if defined(NTDDI_WIN10_RS2) && defined(_WIN32_WINNT_WIN10)
#include <dxgi1_6.h>
#else
#include <dxgi1_5.h>
#endif
#include <cuda_d3d11_interop.h>
#include <vector>
#include "util/common.h"

class D3D11Renderer 
{
public:
    D3D11Renderer(HWND& hwnd, uint32_t* width, uint32_t* height);
    ~D3D11Renderer();

    HRESULT initDevice();
    void cleanupDevice();
    HRESULT initTexture();

    void Draw();
    void onResize();
    cudaGraphicsResource_t* getCudaTextureResource();

private:
    HWND& g_hWnd;
    uint32_t* g_WindowWidth;
    uint32_t* g_WindowHeight;

    IDXGIAdapter* g_pCudaCapableAdapter = NULL;  // Adapter to use
    ID3D11Device* g_pd3dDevice = NULL;           // Our rendering device
    ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
    IDXGISwapChain* g_pSwapChain = NULL;  // The swap chain of the window
    ID3D11RenderTargetView* g_pSwapChainRTV = NULL;  // The Render target view on the swap chain ( used for clear)
    ID3D11RasterizerState* g_pRasterState = NULL;

    ID3D11InputLayout* g_pInputLayout = NULL;

    struct ConstantBuffer {
        float vQuadRect[4];
    };

    ID3D11VertexShader* g_pVertexShader = NULL;
    ID3D11PixelShader* g_pPixelShader = NULL;
    ID3D11Buffer* g_pConstantBuffer = NULL;
    ID3D11SamplerState* g_pSamplerState = NULL;

    struct {
        ID3D11Texture2D* pTexture = NULL;
        ID3D11ShaderResourceView* pSRView = NULL;
        cudaGraphicsResource_t cudaResource = NULL;
        int offsetInShader;
    } g_texture_2d;
};