import React, { useState } from 'react';
import { Trash2, Edit2, Plus, GripVertical, Save, X } from 'lucide-react';

const LLMPipelineApp = () => {
  const [activeTab, setActiveTab] = useState('models');
  const [models, setModels] = useState([]);
  const [prompts, setPrompts] = useState([]);
  const [pipelines, setPipelines] = useState([]);
  const [currentPipeline, setCurrentPipeline] = useState(null);
  const [showPipelineForm, setShowPipelineForm] = useState(false);
  
  const [pipeline, setPipeline] = useState({
    id: null,
    name: '',
    description: '',
    dataSource: '',
    selectedModels: [],
    technicalPrompt: null,
    clinicalPrompt: null,
    antiHallucination: {
      ensemble: false,
      ensembleSize: 3,
      chainOfVerification: false,
      contextualGrounding: false,
      llmAsJudge: false,
      judgeModel: null
    }
  });

  const [editingModel, setEditingModel] = useState(null);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [showModelForm, setShowModelForm] = useState(false);
  const [showPromptForm, setShowPromptForm] = useState(false);

  const [modelForm, setModelForm] = useState({
    id: null,
    name: '',
    description: '',
    provider: 'anthropic',
    model: '',
    temperature: 0.7,
    maxTokens: 2048,
    topP: 1.0
  });

  const [promptForm, setPromptForm] = useState({
    id: null,
    name: '',
    description: '',
    type: 'clinical',
    content: ''
  });

  const showError = (message, error) => {
    console.error(message, error);
    if (typeof window !== 'undefined') {
      window.alert(message);
    }
  };

  const normalizePipelineState = (payload) => ({
    id: payload?.id ?? null,
    name: payload?.name ?? '',
    description: payload?.description ?? '',
    dataSource: payload?.dataSource ?? '',
    selectedModels: Array.isArray(payload?.selectedModels) ? payload.selectedModels : [],
    technicalPrompt: payload?.technicalPrompt ?? null,
    clinicalPrompt: payload?.clinicalPrompt ?? null,
    antiHallucination: {
      ensemble: Boolean(payload?.antiHallucination?.ensemble ?? false),
      ensembleSize: Number(payload?.antiHallucination?.ensembleSize ?? 3) || 3,
      chainOfVerification: Boolean(payload?.antiHallucination?.chainOfVerification ?? false),
      contextualGrounding: Boolean(payload?.antiHallucination?.contextualGrounding ?? false),
      llmAsJudge: Boolean(payload?.antiHallucination?.llmAsJudge ?? false),
      judgeModel: payload?.antiHallucination?.judgeModel ?? null
    }
  });

  const createPipelinePayload = (state) => ({
    name: state.name,
    description: state.description,
    dataSource: state.dataSource,
    selectedModels: state.selectedModels,
    technicalPrompt: state.technicalPrompt,
    clinicalPrompt: state.clinicalPrompt,
    antiHallucination: {
      ensemble: state.antiHallucination.ensemble,
      ensembleSize: state.antiHallucination.ensembleSize,
      chainOfVerification: state.antiHallucination.chainOfVerification,
      contextualGrounding: state.antiHallucination.contextualGrounding,
      llmAsJudge: state.antiHallucination.llmAsJudge,
      judgeModel: state.antiHallucination.llmAsJudge ? state.antiHallucination.judgeModel : null
    }
  });

  const refreshModels = async () => {
    try {
      const data = await jsonRequest('/models');
      setModels(data);
      setPipeline((prev) => {
        if (!prev) {
          return prev;
        }
        const validModelIds = new Set(data.map((item) => item.id));
        const updatedAnti = {
          ...prev.antiHallucination,
          judgeModel: prev.antiHallucination.judgeModel && validModelIds.has(prev.antiHallucination.judgeModel)
            ? prev.antiHallucination.judgeModel
            : null
        };
        return {
          ...prev,
          selectedModels: prev.selectedModels.filter((id) => validModelIds.has(id)),
          antiHallucination: updatedAnti
        };
      });
      setCurrentPipeline((prev) => {
        if (!prev) {
          return prev;
        }
        const validModelIds = new Set(data.map((item) => item.id));
        const updatedAnti = {
          ...prev.antiHallucination,
          judgeModel: prev.antiHallucination.judgeModel && validModelIds.has(prev.antiHallucination.judgeModel)
            ? prev.antiHallucination.judgeModel
            : null
        };
        return {
          ...prev,
          selectedModels: prev.selectedModels.filter((id) => validModelIds.has(id)),
          antiHallucination: updatedAnti
        };
      });
    } catch (error) {
      showError("Erreur lors de l'actualisation des modèles.", error);
    }
  };

  const refreshPrompts = async () => {
    try {
      const data = await jsonRequest('/prompts');
      setPrompts(data);
      setPipeline((prev) => {
        if (!prev) {
          return prev;
        }
        const validPromptIds = new Set(data.map((item) => item.id));
        return {
          ...prev,
          technicalPrompt: prev.technicalPrompt && validPromptIds.has(prev.technicalPrompt) ? prev.technicalPrompt : null,
          clinicalPrompt: prev.clinicalPrompt && validPromptIds.has(prev.clinicalPrompt) ? prev.clinicalPrompt : null
        };
      });
      setCurrentPipeline((prev) => {
        if (!prev) {
          return prev;
        }
        const validPromptIds = new Set(data.map((item) => item.id));
        return {
          ...prev,
          technicalPrompt: prev.technicalPrompt && validPromptIds.has(prev.technicalPrompt) ? prev.technicalPrompt : null,
          clinicalPrompt: prev.clinicalPrompt && validPromptIds.has(prev.clinicalPrompt) ? prev.clinicalPrompt : null
        };
      });
    } catch (error) {
      showError("Erreur lors de l'actualisation des prompts.", error);
    }
  };

  const refreshPipelines = async () => {
    try {
      const data = await jsonRequest('/pipelines');
      setPipelines(data);
      setCurrentPipeline((prev) => {
        if (!prev) {
          return prev;
        }
        const match = data.find((item) => item.id === prev.id);
        return match || null;
      });
    } catch (error) {
      showError("Erreur lors de l'actualisation des pipelines.", error);
    }
  };

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [modelsData, promptsData, pipelinesData] = await Promise.all([
          jsonRequest('/models'),
          jsonRequest('/prompts'),
          jsonRequest('/pipelines')
        ]);
        setModels(modelsData);
        setPrompts(promptsData);
        setPipelines(pipelinesData);
      } catch (error) {
        showError("Erreur lors du chargement initial des données.", error);
      }
    };
    bootstrap();
  }, []);

  // Model Management
  const handleSaveModel = async () => {
    const payload = {
      name: modelForm.name,
      description: modelForm.description,
      provider: modelForm.provider,
      model: modelForm.model,
      temperature: modelForm.temperature,
      maxTokens: modelForm.maxTokens,
      topP: modelForm.topP
    };

    try {
      if (modelForm.id) {
        const updated = await jsonRequest(`/models/${modelForm.id}`, {
          method: 'PUT',
          body: JSON.stringify(payload)
        });
        setModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
        setPipeline((prev) => {
          if (!prev) {
            return prev;
          }
          const validModelIds = new Set([...prev.selectedModels, updated.id]);
          const updatedAnti = {
            ...prev.antiHallucination,
            judgeModel:
              prev.antiHallucination.judgeModel === updated.id ? updated.id : prev.antiHallucination.judgeModel
          };
          return {
            ...prev,
            selectedModels: prev.selectedModels.filter((id) => validModelIds.has(id)),
            antiHallucination: updatedAnti
          };
        });
      } else {
        const created = await jsonRequest('/models', {
          method: 'POST',
          body: JSON.stringify(payload)
        });
        setModels((prev) => [...prev, created]);
      }
      resetModelForm();
      await refreshModels();
      await refreshPipelines();
    } catch (error) {
      showError("Erreur lors de la sauvegarde du modèle.", error);
    }
  };

  const handleEditModel = (model) => {
    setModelForm(model);
    setShowModelForm(true);
  };

  const handleDeleteModel = async (id) => {
    try {
      await jsonRequest(`/models/${id}`, { method: 'DELETE' });
      await refreshModels();
      await refreshPipelines();
    } catch (error) {
      showError("Erreur lors de la suppression du modèle.", error);
    }
  };

  const resetModelForm = () => {
    setModelForm({
      id: null,
      name: '',
      description: '',
      provider: 'anthropic',
      model: '',
      temperature: 0.7,
      maxTokens: 2048,
      topP: 1.0
    });
    setShowModelForm(false);
  };

  // Prompt Management
  const handleSavePrompt = async () => {
    const payload = {
      name: promptForm.name,
      description: promptForm.description,
      type: promptForm.type,
      content: promptForm.content
    };

    try {
      if (promptForm.id) {
        const updated = await jsonRequest(`/prompts/${promptForm.id}`, {
          method: 'PUT',
          body: JSON.stringify(payload)
        });
        setPrompts((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
      } else {
        const created = await jsonRequest('/prompts', {
          method: 'POST',
          body: JSON.stringify(payload)
        });
        setPrompts((prev) => [...prev, created]);
      }
      resetPromptForm();
      await refreshPrompts();
      await refreshPipelines();
    } catch (error) {
      showError("Erreur lors de la sauvegarde du prompt.", error);
    }
  };

  const handleEditPrompt = (prompt) => {
    setPromptForm(prompt);
    setShowPromptForm(true);
  };

  const handleDeletePrompt = async (id) => {
    try {
      await jsonRequest(`/prompts/${id}`, { method: 'DELETE' });
      await refreshPrompts();
      await refreshPipelines();
    } catch (error) {
      showError("Erreur lors de la suppression du prompt.", error);
    }
  };

  const resetPromptForm = () => {
    setPromptForm({
      id: null,
      name: '',
      description: '',
      type: 'clinical',
      content: ''
    });
    setShowPromptForm(false);
  };

  // Pipeline Management
  const handleSavePipeline = async () => {
    const payload = createPipelinePayload(pipeline);

    try {
      let savedPipeline = null;
      if (pipeline.id) {
        savedPipeline = await jsonRequest(`/pipelines/${pipeline.id}`, {
          method: 'PUT',
          body: JSON.stringify(payload)
        });
      } else {
        savedPipeline = await jsonRequest('/pipelines', {
          method: 'POST',
          body: JSON.stringify(payload)
        });
      }
      const normalized = normalizePipelineState(savedPipeline);
      setPipeline(normalized);
      setCurrentPipeline(savedPipeline);
      setShowPipelineForm(false);
      await refreshPipelines();
    } catch (error) {
      showError("Erreur lors de la sauvegarde du pipeline.", error);
    }
  };

  const handleEditPipeline = (pipelineToEdit) => {
    setPipeline(normalizePipelineState(pipelineToEdit));
    setCurrentPipeline(pipelineToEdit);
    setShowPipelineForm(true);
  };

  const handleDeletePipeline = async (id) => {
    try {
      await jsonRequest(`/pipelines/${id}`, { method: 'DELETE' });
      setCurrentPipeline((prev) => (prev && prev.id === id ? null : prev));
      setPipeline((prev) => (prev.id === id ? normalizePipelineState({}) : prev));
      await refreshPipelines();
    } catch (error) {
      showError("Erreur lors de la suppression du pipeline.", error);
    }
  };

  const handleNewPipeline = () => {
    setPipeline(normalizePipelineState({}));
    setShowPipelineForm(true);
  };

  const resetPipelineForm = () => {
    setPipeline(currentPipeline ? normalizePipelineState(currentPipeline) : normalizePipelineState({}));
    setShowPipelineForm(false);
  };

  const loadPipeline = (pipelineId) => {
    const pipelineToLoad = pipelines.find(p => p.id === pipelineId);
    if (!pipelineToLoad) {
      return;
    }
    setPipeline(normalizePipelineState(pipelineToLoad));
    setCurrentPipeline(pipelineToLoad);
    setShowPipelineForm(true);
  };

  const toggleModelInPipeline = (modelId) => {
    setPipeline((prev) => {
      if (prev.selectedModels.includes(modelId)) {
        const nextSelected = prev.selectedModels.filter(id => id !== modelId);
        const shouldClearJudge = prev.antiHallucination.judgeModel === modelId;
        return {
          ...prev,
          selectedModels: nextSelected,
          antiHallucination: {
            ...prev.antiHallucination,
            judgeModel: shouldClearJudge ? null : prev.antiHallucination.judgeModel
          }
        };
      }
      return {
        ...prev,
        selectedModels: [...prev.selectedModels, modelId]
      };
    });
  };

  const getModelById = (id) => models.find(m => m.id === id);
  const getPromptById = (id) => prompts.find(p => p.id === id);

  const renderModelsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Configuration des Modèles</h2>
        <button
          onClick={() => setShowModelForm(true)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus size={20} />
          Nouveau Modèle
        </button>
      </div>

      {showModelForm && (
        <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-blue-200">
          <h3 className="text-xl font-semibold mb-4">
            {modelForm.id ? 'Modifier le Modèle' : 'Nouveau Modèle'}
          </h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Nom</label>
              <input
                type="text"
                value={modelForm.name}
                onChange={(e) => setModelForm({ ...modelForm, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Mon modèle GPT-4"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Provider</label>
              <select
                value={modelForm.provider}
                onChange={(e) => setModelForm({ ...modelForm, provider: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="anthropic">Anthropic</option>
                <option value="ollama">Ollama</option>
                <option value="openai">OpenAI</option>
              </select>
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
              <textarea
                value={modelForm.description}
                onChange={(e) => setModelForm({ ...modelForm, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="2"
                placeholder="Description du modèle..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Modèle</label>
              <input
                type="text"
                value={modelForm.model}
                onChange={(e) => setModelForm({ ...modelForm, model: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="claude-3-sonnet, llama2, gpt-4..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Température: {modelForm.temperature}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={modelForm.temperature}
                onChange={(e) => setModelForm({ ...modelForm, temperature: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Tokens</label>
              <input
                type="number"
                value={modelForm.maxTokens}
                onChange={(e) => setModelForm({ ...modelForm, maxTokens: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Top P: {modelForm.topP}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={modelForm.topP}
                onChange={(e) => setModelForm({ ...modelForm, topP: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>

          <div className="flex gap-3 mt-6">
            <button
              onClick={handleSaveModel}
              className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
            >
              <Save size={18} />
              Enregistrer
            </button>
            <button
              onClick={resetModelForm}
              className="flex items-center gap-2 bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
            >
              <X size={18} />
              Annuler
            </button>
          </div>
        </div>
      )}

      <div className="grid gap-4">
        {models.map(model => (
          <div key={model.id} className="bg-white p-5 rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-200">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-800">{model.name}</h3>
                <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                <div className="flex gap-4 mt-3 text-sm">
                  <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
                    {model.provider}
                  </span>
                  <span className="text-gray-600">Modèle: {model.model}</span>
                  <span className="text-gray-600">Temp: {model.temperature}</span>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleEditModel(model)}
                  className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  <Edit2 size={18} />
                </button>
                <button
                  onClick={() => handleDeleteModel(model.id)}
                  className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {models.length === 0 && !showModelForm && (
        <div className="text-center py-12 text-gray-500">
          Aucun modèle configuré. Cliquez sur "Nouveau Modèle" pour commencer.
        </div>
      )}
    </div>
  );

  const renderPromptsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Configuration des Prompts</h2>
        <button
          onClick={() => setShowPromptForm(true)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus size={20} />
          Nouveau Prompt
        </button>
      </div>

      {showPromptForm && (
        <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-blue-200">
          <h3 className="text-xl font-semibold mb-4">
            {promptForm.id ? 'Modifier le Prompt' : 'Nouveau Prompt'}
          </h3>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Nom</label>
                <input
                  type="text"
                  value={promptForm.name}
                  onChange={(e) => setPromptForm({ ...promptForm, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Extraction des diagnostics"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
                <select
                  value={promptForm.type}
                  onChange={(e) => setPromptForm({ ...promptForm, type: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                >
                  <option value="clinical">Prompt Clinique</option>
                  <option value="technical">Prompt Technique</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
              <textarea
                value={promptForm.description}
                onChange={(e) => setPromptForm({ ...promptForm, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="2"
                placeholder="Description du prompt..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Contenu du Prompt</label>
              <textarea
                value={promptForm.content}
                onChange={(e) => setPromptForm({ ...promptForm, content: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
                rows="10"
                placeholder="Entrez le contenu de votre prompt ici..."
              />
            </div>
          </div>

          <div className="flex gap-3 mt-6">
            <button
              onClick={handleSavePrompt}
              className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
            >
              <Save size={18} />
              Enregistrer
            </button>
            <button
              onClick={resetPromptForm}
              className="flex items-center gap-2 bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
            >
              <X size={18} />
              Annuler
            </button>
          </div>
        </div>
      )}

      <div className="grid gap-4">
        {prompts.map(prompt => (
          <div key={prompt.id} className="bg-white p-5 rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-200">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold text-gray-800">{prompt.name}</h3>
                  <span className={`text-xs px-3 py-1 rounded-full ${
                    prompt.type === 'clinical' 
                      ? 'bg-purple-100 text-purple-800' 
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {prompt.type === 'clinical' ? 'Clinique' : 'Technique'}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{prompt.description}</p>
                <div className="mt-3 p-3 bg-gray-50 rounded border border-gray-200 text-sm font-mono text-gray-700 max-h-24 overflow-y-auto">
                  {prompt.content || 'Aucun contenu'}
                </div>
              </div>
              <div className="flex gap-2 ml-4">
                <button
                  onClick={() => handleEditPrompt(prompt)}
                  className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  <Edit2 size={18} />
                </button>
                <button
                  onClick={() => handleDeletePrompt(prompt.id)}
                  className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {prompts.length === 0 && !showPromptForm && (
        <div className="text-center py-12 text-gray-500">
          Aucun prompt configuré. Cliquez sur "Nouveau Prompt" pour commencer.
        </div>
      )}
    </div>
  );

  const renderPipelineTab = () => {
    const clinicalPrompts = prompts.filter(p => p.type === 'clinical');
    const technicalPrompts = prompts.filter(p => p.type === 'technical');

    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-800">Gestion des Pipelines</h2>
          <button
            onClick={handleNewPipeline}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus size={20} />
            Nouveau Pipeline
          </button>
        </div>

        {/* Liste des pipelines existants */}
        {pipelines.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">Pipelines Enregistrés</h3>
            <div className="grid gap-3">
              {pipelines.map(p => (
                <div key={p.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800">{p.name}</h4>
                    <p className="text-sm text-gray-600">{p.description}</p>
                    <div className="flex gap-3 mt-2 text-xs text-gray-500">
                      <span>{p.selectedModels.length} modèle(s)</span>
                      <span>•</span>
                      <span>{Object.values(p.antiHallucination).filter(v => v === true).length} stratégie(s)</span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => loadPipeline(p.id)}
                      className="px-3 py-2 text-sm bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors"
                    >
                      Charger
                    </button>
                    <button
                      onClick={() => handleEditPipeline(p)}
                      className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    >
                      <Edit2 size={18} />
                    </button>
                    <button
                      onClick={() => handleDeletePipeline(p.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Formulaire de configuration du pipeline */}
        {showPipelineForm && (
          <>
            <div className="bg-white p-6 rounded-lg shadow-md border-2 border-blue-200">
              <h3 className="text-xl font-semibold mb-4">
                {pipeline.id ? 'Modifier le Pipeline' : 'Nouveau Pipeline'}
              </h3>
              
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Nom du Pipeline</label>
                  <input
                    type="text"
                    value={pipeline.name}
                    onChange={(e) => setPipeline({ ...pipeline, name: e.target.value })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Pipeline extraction diagnostics"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                  <input
                    type="text"
                    value={pipeline.description}
                    onChange={(e) => setPipeline({ ...pipeline, description: e.target.value })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Description du pipeline..."
                  />
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">Source de Données</h3>
              <input
                type="text"
                value={pipeline.dataSource}
                onChange={(e) => setPipeline({ ...pipeline, dataSource: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Chemin vers le dataset ou source de données..."
              />
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">Sélection des Modèles</h3>
              {models.length === 0 ? (
                <p className="text-gray-500">Aucun modèle disponible. Configurez des modèles dans l'onglet correspondant.</p>
              ) : (
                <div className="space-y-2">
                  {models.map(model => (
                    <label key={model.id} className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                      <input
                        type="checkbox"
                        checked={pipeline.selectedModels.includes(model.id)}
                        onChange={() => toggleModelInPipeline(model.id)}
                        className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                      />
                      <div className="flex-1">
                        <div className="font-medium text-gray-800">{model.name}</div>
                        <div className="text-sm text-gray-600">{model.provider} - {model.model}</div>
                      </div>
                    </label>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Prompt Technique</h3>
                <p className="text-sm text-gray-600 mb-3">Format de sortie (JSON, etc.)</p>
                <select
                  value={pipeline.technicalPrompt || ''}
                  onChange={(e) => setPipeline({ ...pipeline, technicalPrompt: e.target.value ? parseInt(e.target.value) : null })}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Aucun prompt technique</option>
                  {technicalPrompts.map(prompt => (
                    <option key={prompt.id} value={prompt.id}>{prompt.name}</option>
                  ))}
                </select>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Prompt Clinique</h3>
                <p className="text-sm text-gray-600 mb-3">Définitions médicales à extraire</p>
                <select
                  value={pipeline.clinicalPrompt || ''}
                  onChange={(e) => setPipeline({ ...pipeline, clinicalPrompt: e.target.value ? parseInt(e.target.value) : null })}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Aucun prompt clinique</option>
                  {clinicalPrompts.map(prompt => (
                    <option key={prompt.id} value={prompt.id}>{prompt.name}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">Stratégies Anti-Hallucination</h3>
              
              <div className="space-y-4">
                <label className="flex items-start gap-3 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={pipeline.antiHallucination.ensemble}
                    onChange={(e) => setPipeline({
                      ...pipeline,
                      antiHallucination: { ...pipeline.antiHallucination, ensemble: e.target.checked }
                    })}
                    className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 mt-1"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-800">Ensemble Methods</div>
                    <p className="text-sm text-gray-600 mt-1">
                      Utiliser plusieurs modèles en parallèle et ne conserver que les concepts identifiés de manière cohérente
                    </p>
                    {pipeline.antiHallucination.ensemble && (
                      <div className="mt-3">
                        <label className="text-sm text-gray-700">Taille de l'ensemble: {pipeline.antiHallucination.ensembleSize}</label>
                        <input
                          type="range"
                          min="2"
                          max="10"
                          value={pipeline.antiHallucination.ensembleSize}
                          onChange={(e) => setPipeline({
                            ...pipeline,
                            antiHallucination: { ...pipeline.antiHallucination, ensembleSize: parseInt(e.target.value) }
                          })}
                          className="w-full mt-1"
                        />
                      </div>
                    )}
                  </div>
                </label>

                <label className="flex items-start gap-3 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={pipeline.antiHallucination.chainOfVerification}
                    onChange={(e) => setPipeline({
                      ...pipeline,
                      antiHallucination: { ...pipeline.antiHallucination, chainOfVerification: e.target.checked }
                    })}
                    className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 mt-1"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-800">Chain-of-Verification (CoVe)</div>
                    <p className="text-sm text-gray-600 mt-1">
                      Permettre au modèle de vérifier ses propres extractions par auto-validation
                    </p>
                  </div>
                </label>

                <label className="flex items-start gap-3 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={pipeline.antiHallucination.contextualGrounding}
                    onChange={(e) => setPipeline({
                      ...pipeline,
                      antiHallucination: { ...pipeline.antiHallucination, contextualGrounding: e.target.checked }
                    })}
                    className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 mt-1"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-800">Contextual Grounding</div>
                    <p className="text-sm text-gray-600 mt-1">
                      S'assurer que les concepts extraits sont supportés par le texte source via des vérifications d'implication
                    </p>
                  </div>
                </label>

                <label className="flex items-start gap-3 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={pipeline.antiHallucination.llmAsJudge}
                    onChange={(e) => setPipeline({
                      ...pipeline,
                      antiHallucination: { ...pipeline.antiHallucination, llmAsJudge: e.target.checked }
                    })}
                    className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500 mt-1"
                  />
                  <div className="flex-1">
                    <div className="font-medium text-gray-800">LLM-as-a-Judge</div>
                    <p className="text-sm text-gray-600 mt-1">
                      Utiliser un modèle indépendant pour évaluer la validité des extractions
                    </p>
                    {pipeline.antiHallucination.llmAsJudge && (
                      <div className="mt-3">
                        <label className="text-sm text-gray-700 block mb-1">Modèle juge:</label>
                        <select
                          value={pipeline.antiHallucination.judgeModel || ''}
                          onChange={(e) => setPipeline({
                            ...pipeline,
                            antiHallucination: { 
                              ...pipeline.antiHallucination, 
                              judgeModel: e.target.value ? parseInt(e.target.value) : null 
                            }
                          })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="">Sélectionner un modèle</option>
                          {models.map(model => (
                            <option key={model.id} value={model.id}>{model.name}</option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                </label>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleSavePipeline}
                className="flex items-center gap-2 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors"
              >
                <Save size={20} />
                Enregistrer le Pipeline
              </button>
              <button
                onClick={resetPipelineForm}
                className="flex items-center gap-2 bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 transition-colors"
              >
                <X size={20} />
                Annuler
              </button>
            </div>
          </>
        )}

        {/* Visualisation du pipeline actuel */}
        {(currentPipeline || (pipeline.selectedModels.length > 0 && showPipelineForm)) && (
          <div className="bg-white p-8 rounded-lg shadow-lg border-2 border-blue-300">
            <h3 className="text-2xl font-bold mb-6 text-gray-800">Synthèse Visuelle du Pipeline</h3>
            
            <div className="space-y-6">
              {/* Modèles LLM */}
              <div>
                <h4 className="text-sm font-semibold text-gray-600 mb-3">MODÈLES LLM PARALLÈLES</h4>
                <div className="flex gap-4 flex-wrap">
                  {pipeline.selectedModels.map((modelId, idx) => {
                    const model = getModelById(modelId);
                    return model ? (
                      <div key={modelId} className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-5 rounded-xl shadow-lg flex-1 min-w-[200px]">
                        <div className="flex items-center justify-center mb-3">
                          <div className="w-12 h-12 bg-white bg-opacity-30 rounded-full flex items-center justify-center">
                            <span className="text-xl font-bold">L{idx + 1}</span>
                          </div>
                        </div>
                        <h5 className="font-bold text-center text-lg mb-2">{model.name}</h5>
                        <div className="text-center space-y-1">
                          <div className="bg-white bg-opacity-20 rounded px-2 py-1 text-sm">
                            Medical Concept Extraction
                          </div>
                          {pipeline.clinicalPrompt && (
                            <div className="bg-orange-500 bg-opacity-90 rounded px-2 py-1 text-xs">
                              Contextual Grounding
                            </div>
                          )}
                        </div>
                      </div>
                    ) : null;
                  })}
                </div>
              </div>

              {/* Chain of Verification */}
              {pipeline.antiHallucination.chainOfVerification && (
                <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white p-5 rounded-xl shadow-lg">
                  <h5 className="font-bold text-center text-lg mb-2">Chain-of-Verification (CoVe)</h5>
                  <p className="text-center text-sm opacity-90">
                    Self-checking: models validate their own extractions
                  </p>
                  <p className="text-center text-xs mt-2 opacity-75">
                    Prompt each LLM to verify consistency and grounding
                  </p>
                </div>
              )}

              {/* LLM-as-a-Judge */}
              {pipeline.antiHallucination.llmAsJudge && pipeline.antiHallucination.judgeModel && (
                <div className="bg-gradient-to-r from-teal-600 to-teal-700 text-white p-5 rounded-xl shadow-lg">
                  <div className="flex items-center justify-center mb-3">
                    <div className="w-12 h-12 bg-white bg-opacity-30 rounded-full flex items-center justify-center">
                      <span className="text-xl">⚖️</span>
                    </div>
                  </div>
                  <h5 className="font-bold text-center text-lg mb-2">LLM-as-a-Judge</h5>
                  <p className="text-center text-sm opacity-90">
                    Independent model evaluates extraction validity
                  </p>
                  <p className="text-center text-xs mt-2 opacity-75">
                    Assesses against source text and reasoning context
                  </p>
                  <div className="text-center mt-3">
                    <span className="bg-white bg-opacity-20 rounded px-3 py-1 text-sm">
                      {getModelById(pipeline.antiHallucination.judgeModel)?.name || 'Judge Model'}
                    </span>
                  </div>
                </div>
              )}

              {/* Consensus Mechanism */}
              {pipeline.antiHallucination.ensemble && (
                <div className="bg-gradient-to-r from-green-600 to-green-700 text-white p-6 rounded-xl shadow-lg">
                  <h5 className="font-bold text-center text-lg mb-2">Consensus Mechanism</h5>
                  <p className="text-center text-sm opacity-90 mb-4">
                    Retain only concepts identified reliably across multiple models
                  </p>
                  <div className="flex justify-center items-center gap-3">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-white bg-opacity-30 rounded-full flex items-center justify-center mb-2">
                        <span className="text-2xl font-bold">{pipeline.antiHallucination.ensembleSize}</span>
                      </div>
                      <p className="text-xs">Ensemble Size</p>
                    </div>
                  </div>
                  <div className="mt-4 bg-green-800 bg-opacity-40 rounded-lg p-3">
                    <h6 className="font-semibold text-sm mb-2">Shared Validated Concepts</h6>
                    <p className="text-xs opacity-90">
                      Validated across {pipeline.antiHallucination.ensembleSize} models with consensus
                    </p>
                  </div>
                </div>
              )}

              {/* Prompts utilisés */}
              <div className="grid grid-cols-2 gap-4">
                {pipeline.clinicalPrompt && (
                  <div className="bg-purple-50 border-2 border-purple-300 p-4 rounded-lg">
                    <h5 className="font-semibold text-purple-800 mb-2">Prompt Clinique</h5>
                    <p className="text-sm text-purple-700">
                      {getPromptById(pipeline.clinicalPrompt)?.name || 'N/A'}
                    </p>
                  </div>
                )}
                {pipeline.technicalPrompt && (
                  <div className="bg-green-50 border-2 border-green-300 p-4 rounded-lg">
                    <h5 className="font-semibold text-green-800 mb-2">Prompt Technique</h5>
                    <p className="text-sm text-green-700">
                      {getPromptById(pipeline.technicalPrompt)?.name || 'N/A'}
                    </p>
                  </div>
                )}
              </div>

              {/* Statistiques */}
              <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-5 rounded-lg border border-indigo-200">
                <h4 className="font-semibold text-gray-800 mb-3">Résumé de Configuration</h4>
                <div className="grid grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-blue-600">{pipeline.selectedModels.length}</div>
                    <div className="text-xs text-gray-600">Modèles</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-600">
                      {Object.values(pipeline.antiHallucination).filter(v => v === true).length}
                    </div>
                    <div className="text-xs text-gray-600">Stratégies</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-600">
                      {pipeline.clinicalPrompt ? '✓' : '✗'}
                    </div>
                    <div className="text-xs text-gray-600">Clinique</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-teal-600">
                      {pipeline.technicalPrompt ? '✓' : '✗'}
                    </div>
                    <div className="text-xs text-gray-600">Technique</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto p-6">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            LLM Medical Pipeline Configuration
          </h1>
          <p className="text-gray-600">
            Configuration de pipeline pour l'extraction d'informations cliniques avec stratégies anti-hallucination
          </p>
        </header>

        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('models')}
              className={`flex-1 px-6 py-4 font-medium transition-colors ${
                activeTab === 'models'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Configuration des Modèles
            </button>
            <button
              onClick={() => setActiveTab('prompts')}
              className={`flex-1 px-6 py-4 font-medium transition-colors ${
                activeTab === 'prompts'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Configuration des Prompts
            </button>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`flex-1 px-6 py-4 font-medium transition-colors ${
                activeTab === 'pipeline'
                  ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              Pipeline
            </button>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-6">
          {activeTab === 'models' && renderModelsTab()}
          {activeTab === 'prompts' && renderPromptsTab()}
          {activeTab === 'pipeline' && renderPipelineTab()}
        </div>
      </div>
    </div>
  );
};

export default LLMPipelineApp;
