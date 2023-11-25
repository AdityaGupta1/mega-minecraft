#include "d3d11Renderer.h"

static const char g_simpleShaders[] =
"cbuffer cbuf \n"
"{ \n"
"  float4 g_vQuadRect; \n"
"} \n"
"Texture2D g_Texture2D; \n"
"\n"
"SamplerState samLinear{ \n"
"    Filter = MIN_MAG_LINEAR_MIP_POINT; \n"
"};\n"
"\n"
"struct Fragment{ \n"
"    float4 Pos : SV_POSITION;\n"
"    float3 Tex : TEXCOORD0; };\n"
"\n"
"float3 ACESFilm(float3 x) {\n"
"    float a = 2.51f;\n"
"    float b = 0.03f;\n"
"    float c = 2.43f;\n"
"    float d = 0.59f;\n"
"    float e = 0.14f;\n"
"    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);\n"
"}\n"
"\n"
"Fragment VS( uint vertexId : SV_VertexID )\n"
"{\n"
"    Fragment f;\n"
"    f.Tex = float3( 0.f, 0.f, 0.f); \n"
"    if (vertexId == 1) f.Tex.x = 1.f; \n"
"    else if (vertexId == 2) f.Tex.y = 1.f; \n"
"    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f); \n"
"    \n"
"    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n"
"    \n"
"    return f;\n"
"}\n"
"\n"
"float4 PS( Fragment f ) : SV_Target\n"
"{\n"
"    float3 out_col = g_Texture2D.Sample( samLinear, f.Tex.xy ).rgb;\n"
"    return float4(pow(ACESFilm(out_col), float3(0.45454545454545454545454545455f, 0.45454545454545454545454545455f, 0.45454545454545454545454545455f)), 1.f);\n"
"}\n"
"\n";

D3D11Renderer::D3D11Renderer(HWND& hWnd, uint32_t* width, uint32_t* height)
    : g_hWnd(hWnd), g_WindowWidth(width), g_WindowHeight(height) 
{
    if (FAILED(initDevice()) || FAILED(initTexture())) {
        exit(1);
    }
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource, g_texture_2d.pTexture, cudaGraphicsRegisterFlagsNone));
}

D3D11Renderer::~D3D11Renderer() {
    cleanupDevice();
}

cudaGraphicsResource_t* D3D11Renderer::getCudaTextureResource()
{
    return &g_texture_2d.cudaResource;
}

HRESULT D3D11Renderer::initDevice()
{
    HRESULT hr = S_OK;
    cudaError_t cuStatus = cudaSuccess;

    IDXGIAdapter* pAdapter;
    std::vector<IDXGIAdapter*> vAdapters;
    IDXGIFactory1* pFactory = NULL;

    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
    if (FAILED(hr)) {
        fprintf(stderr, "CreateDXGIFactory failed with error code %ld", hr);
        return E_FAIL;
    }

    for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        int cuDevice;
        cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
        cudaGetLastError();
        if (cuStatus == cudaSuccess) {
            vAdapters.push_back(pAdapter);
        }
    }

    g_pCudaCapableAdapter = vAdapters[0];

    if (pFactory) {
        pFactory->Release();
    }

    // Set up the structure used to create the device and swapchain
    D3D_FEATURE_LEVEL tour_fl[] = { D3D_FEATURE_LEVEL_11_0, 
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    D3D_FEATURE_LEVEL flRes;
    // Create device and swapchain
    //hr = D3D11CreateDeviceAndSwapChain(
    //    g_pCudaCapableAdapter,
    //    D3D_DRIVER_TYPE_UNKNOWN,  // D3D_DRIVER_TYPE_HARDWARE,
    //    NULL,  // HMODULE Software
    //    0,  // UINT Flags
    //    tour_fl,  // D3D_FEATURE_LEVEL* pFeatureLevels
    //    3,  // FeatureLevels
    //    D3D11_SDK_VERSION,  // UINT SDKVersion
    //    &sd,  // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
    //    &g_pSwapChain,  // IDXGISwapChain** ppSwapChain
    //    &g_pd3dDevice,  // ID3D11Device** ppDevice
    //    &flRes,  // D3D_FEATURE_LEVEL* pFeatureLevel
    //    &g_pd3dDeviceContext  // ID3D11DeviceContext** ppImmediateContext
    //);
    //if (FAILED(hr)) {
    //    fprintf(stderr, "Initializing Device and SwapChain failed with 0x%X\n", hr);
    //    return E_FAIL;
    //}

    hr = D3D11CreateDevice(g_pCudaCapableAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, tour_fl, 3, D3D11_SDK_VERSION, &g_pd3dDevice, &flRes, &g_pd3dDeviceContext);
    if (FAILED(hr)) {
        fprintf(stderr, "Creating Device failed with 0x%X\n", hr);
        return E_FAIL;
    }

    // Obtain DXGI factory from device (since we used nullptr for pAdapter above)
    IDXGIFactory1* dxgiFactory = nullptr;
    {
        IDXGIDevice* dxgiDevice = nullptr;
        hr = g_pd3dDevice->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice));
        if (SUCCEEDED(hr))
        {
            IDXGIAdapter* adapter = nullptr;
            hr = dxgiDevice->GetAdapter(&adapter);
            if (SUCCEEDED(hr))
            {
                hr = adapter->GetParent(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&dxgiFactory));
                adapter->Release();
            }
            dxgiDevice->Release();
        }
    }
    if (FAILED(hr)) {
        fprintf(stderr, "Getting DXGI Factory from Device failed with 0x%X\n", hr);
        return E_FAIL;
    }

    // Create swap chain
    IDXGIFactory2* dxgiFactory2 = nullptr;
    hr = dxgiFactory->QueryInterface(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&dxgiFactory2));
    if (dxgiFactory2)
    {
        // DirectX 11.1 or later
        ID3D11Device1* g_pd3dDevice1;
        ID3D11DeviceContext1* g_pd3dDeviceContext1;
        IDXGISwapChain1* g_pSwapChain1;

        hr = g_pd3dDevice->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&g_pd3dDevice1));
        if (SUCCEEDED(hr))
        {
            (void)g_pd3dDeviceContext->QueryInterface(__uuidof(ID3D11DeviceContext1), reinterpret_cast<void**>(&g_pd3dDeviceContext1));
        }
        else {
            fprintf(stderr, "Getting Id3D11DeviceContext1 failed with 0x%X\n", hr);
        }

        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = *g_WindowWidth;
        sd.Height = *g_WindowHeight;
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.SampleDesc.Quality = 0;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 1;

        hr = dxgiFactory2->CreateSwapChainForHwnd(g_pd3dDevice, g_hWnd, &sd, nullptr, nullptr, &g_pSwapChain1);
        if (SUCCEEDED(hr))
        {
            hr = g_pSwapChain1->QueryInterface(__uuidof(IDXGISwapChain), reinterpret_cast<void**>(&g_pSwapChain));
        }
        else {
            fprintf(stderr, "Creating Swap Chain for Hwnd failed with 0x%X\n", hr);
        }

        dxgiFactory2->Release();
    }
    else
    {
        // DirectX 11.0 systems
        DXGI_SWAP_CHAIN_DESC sd = {};
        sd.BufferCount = 1;
        sd.BufferDesc.Width = *g_WindowWidth;
        sd.BufferDesc.Height = *g_WindowHeight;
        sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.BufferDesc.RefreshRate.Numerator = 60;
        sd.BufferDesc.RefreshRate.Denominator = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.OutputWindow = g_hWnd;
        sd.SampleDesc.Count = 1;
        sd.SampleDesc.Quality = 0;
        sd.Windowed = TRUE;

        hr = dxgiFactory->CreateSwapChain(g_pd3dDevice, &sd, &g_pSwapChain);
    }
    if (FAILED(hr)) {
        fprintf(stdout, "%u %u", *g_WindowWidth, *g_WindowHeight);
        fprintf(stderr, "Creating Swap Chain failed with 0x%X\n", hr);
        return E_FAIL;
    }

    for (IDXGIAdapter* pA : vAdapters) {
        pA->Release();
    }

    // Get the immediate DeviceContext
    g_pd3dDevice->GetImmediateContext(&g_pd3dDeviceContext);

    // Create a render target view of the swapchain
    ID3D11Texture2D* pBuffer;
    hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBuffer);
    if (FAILED(hr)) {
        fprintf(stderr, "Getting Swap Chain Buffer failed with 0x%X\n", hr);
        return E_FAIL;
    }

    hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
    if (FAILED(hr)) {
        fprintf(stderr, "Creating Render Target View failed with 0x%X\n", hr);
        return E_FAIL;
    }

    pBuffer->Release();

    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = *g_WindowWidth;
    vp.Height = *g_WindowHeight;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pd3dDeviceContext->RSSetViewports(1, &vp);

    ID3DBlob* pShader;
    ID3DBlob* pErrorMsgs;
    // Vertex shader
    {
        hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL,
            NULL, "VS", "vs_4_0", 0 /*Flags1*/, 0 /*Flags2*/, &pShader,
            &pErrorMsgs);

        if (FAILED(hr)) {
            const char* pStr = (const char*)pErrorMsgs->GetBufferPointer();
            printf(pStr);
            return E_FAIL;
        }

        hr = g_pd3dDevice->CreateVertexShader(pShader->GetBufferPointer(),
            pShader->GetBufferSize(), NULL,
            &g_pVertexShader);
        if (FAILED(hr)) {
            fprintf(stderr, "Creating Vertex Shader failed with 0x%X\n", hr);
            return E_FAIL;
        }
        // Bind Vertex Shader
        g_pd3dDeviceContext->VSSetShader(g_pVertexShader, NULL, 0);
    }
    // Pixel shader
    {
        hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL,
            NULL, "PS", "ps_4_0", 0 /*Flags1*/, 0 /*Flags2*/, &pShader,
            &pErrorMsgs);

        if (FAILED(hr)) {
            const char* pStr = (const char*)pErrorMsgs->GetBufferPointer();
            printf(pStr);
            return E_FAIL;
        }

        hr = g_pd3dDevice->CreatePixelShader(pShader->GetBufferPointer(),
            pShader->GetBufferSize(), NULL,
            &g_pPixelShader);

        if (FAILED(hr)) {
            fprintf(stderr, "Creating Pixel Shader failed with 0x%X\n", hr);
            return E_FAIL;
        }

        // Bind Pixel Shader
        g_pd3dDeviceContext->PSSetShader(g_pPixelShader, NULL, 0);
    }
    // Create the constant buffer
    {
        D3D11_BUFFER_DESC cbDesc;
        cbDesc.Usage = D3D11_USAGE_DYNAMIC;
        cbDesc.BindFlags =
            D3D11_BIND_CONSTANT_BUFFER;  // D3D11_BIND_SHADER_RESOURCE;
        cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        cbDesc.MiscFlags = 0;
        cbDesc.ByteWidth = 16 * ((sizeof(ConstantBuffer) + 16) / 16);
        // cbDesc.StructureByteStride = 0;
        hr = g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pConstantBuffer);
        if (FAILED(hr)) {
            fprintf(stderr, "Creating Constant Buffer failed with 0x%X\n", hr);
            return E_FAIL;
        }

        float quadRect[4] = { -1.f, -1.f, 2.0f, 2.f };

        HRESULT hr;
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        ConstantBuffer* pcb;
        hr = g_pd3dDeviceContext->Map(g_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD,
            0, &mappedResource);
        pcb = (ConstantBuffer*)mappedResource.pData;
        {
            memcpy(pcb->vQuadRect, quadRect, sizeof(float) * 4);
        }
        g_pd3dDeviceContext->Unmap(g_pConstantBuffer, 0);

        /*ConstantBuffer cb;
        cb.vQuadRect[0] = -1.f;
        cb.vQuadRect[1] = -1.f;
        cb.vQuadRect[2] = 1.f;
        cb.vQuadRect[3] = 1.f;
        g_pd3dDeviceContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);*/

        g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
        g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    }
    // SamplerState
    {
        D3D11_SAMPLER_DESC sDesc;
        sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sDesc.MinLOD = 0;
        sDesc.MaxLOD = 8;
        sDesc.MipLODBias = 0;
        sDesc.MaxAnisotropy = 1;
        hr = g_pd3dDevice->CreateSamplerState(&sDesc, &g_pSamplerState);
        if (FAILED(hr)) {
            fprintf(stderr, "Creating Sampler State failed with 0x%X\n", hr);
            return E_FAIL;
        }
        g_pd3dDeviceContext->PSSetSamplers(0, 1, &g_pSamplerState);
    }
    
    // Setup  no Input Layout
    g_pd3dDeviceContext->IASetInputLayout(0);
    g_pd3dDeviceContext->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    D3D11_RASTERIZER_DESC rasterizerState;
    rasterizerState.FillMode = D3D11_FILL_SOLID;
    rasterizerState.CullMode = D3D11_CULL_FRONT;
    rasterizerState.FrontCounterClockwise = false;
    rasterizerState.DepthBias = false;
    rasterizerState.DepthBiasClamp = 0;
    rasterizerState.SlopeScaledDepthBias = 0;
    rasterizerState.DepthClipEnable = false;
    rasterizerState.ScissorEnable = false;
    rasterizerState.MultisampleEnable = false;
    rasterizerState.AntialiasedLineEnable = false;
    g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
    g_pd3dDeviceContext->RSSetState(g_pRasterState);

    return S_OK;
}

void D3D11Renderer::cleanupDevice()
{
    if (g_texture_2d.cudaResource)
        CUDA_CHECK(cudaGraphicsUnregisterResource(g_texture_2d.cudaResource));

    g_texture_2d.pSRView->Release();
    g_texture_2d.pTexture->Release();

    if (g_pVertexShader)
        g_pVertexShader->Release();
    if (g_pPixelShader)
        g_pPixelShader->Release();
    if (g_pConstantBuffer)
        g_pConstantBuffer->Release();
    if (g_pSamplerState)
        g_pSamplerState->Release();

    if (g_pd3dDeviceContext)
        g_pd3dDeviceContext->ClearState();
    if (g_pSwapChain)
        g_pSwapChain->Release();
    if (g_pSwapChainRTV)
        g_pSwapChainRTV->Release();
    if (g_pRasterState)
        g_pRasterState->Release();
    if (g_pInputLayout)
        g_pInputLayout->Release();
    if (g_pd3dDevice)
        g_pd3dDevice->Release();
    if (g_pCudaCapableAdapter)
        g_pCudaCapableAdapter->Release();
}

HRESULT D3D11Renderer::initTexture() {
    HRESULT hr = S_OK;

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = *g_WindowWidth;
    desc.Height = *g_WindowHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    hr = g_pd3dDevice->CreateTexture2D(&desc, NULL, &g_texture_2d.pTexture);
    if (FAILED(hr)) {
        fprintf(stderr, "Creating 2D Texture failed with %ld", hr);
        return E_FAIL;
    }

    hr = g_pd3dDevice->CreateShaderResourceView(g_texture_2d.pTexture, NULL, &g_texture_2d.pSRView);
    if (FAILED(hr)) {
        fprintf(stderr, "Creating Shader Resource View for 2D texture failed with %ld", hr);
        return E_FAIL;
    }
    g_texture_2d.offsetInShader = 0;  // to be clean we should look for the offset from the shader code
    g_pd3dDeviceContext->PSSetShaderResources(g_texture_2d.offsetInShader, 1,
        &g_texture_2d.pSRView);

    return S_OK;
}

void D3D11Renderer::Draw()
{
    // Clear the backbuffer to a black color
    float ClearColor[4] = { 0.5f, 0.5f, 0.6f, 1.0f };
    g_pd3dDeviceContext->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

    g_pd3dDeviceContext->Draw(4, 0);

    // Present the backbuffer contents to the display
    g_pSwapChain->Present(0, 0);
}