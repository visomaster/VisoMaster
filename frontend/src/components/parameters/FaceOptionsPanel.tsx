import { useState } from 'react'
import { Copy, ClipboardPaste, RotateCcw, Plus, GripVertical, X, ChevronDown, ChevronRight, Pin } from 'lucide-react'
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core'
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy, useSortable, arrayMove } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { useEvents } from '@/hooks/useEvents'
import { cn } from '@/lib/utils'
import { ParameterBlock } from './ParameterBlock'

const ALL_BLOCKS = [
  'Face Similarity',
  'Face Mask',
  'Landmarks Correction',
  'Detection',
  'Swapper',
  'Frame Enhancer',
  'Color Correction',
  'Expression Restorer',
  'Face Editor',
]

function SortableBlock({ id, onRemove }: { id: string; onRemove: () => void }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id })
  const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1 }
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div ref={setNodeRef} style={style} className="bg-zinc-800/50 border border-zinc-700 rounded-lg overflow-hidden">
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-zinc-700/50">
        <button {...attributes} {...listeners} className="text-zinc-600 hover:text-zinc-400 cursor-grab active:cursor-grabbing p-0.5">
          <GripVertical size={13} />
        </button>
        <button onClick={() => setCollapsed(v => !v)} className="flex-1 flex items-center gap-1.5 text-left">
          {collapsed ? <ChevronRight size={12} className="text-zinc-500" /> : <ChevronDown size={12} className="text-zinc-500" />}
          <span className="text-xs font-medium text-zinc-300">{id}</span>
        </button>
        <button onClick={onRemove} className="p-0.5 text-zinc-600 hover:text-red-400 transition-colors">
          <X size={12} />
        </button>
      </div>
      {!collapsed && (
        <div className="p-2">
          <ParameterBlock blockName={id} />
        </div>
      )}
    </div>
  )
}

export function FaceOptionsPanel() {
  const { selectedFaceId, parameters, activeBlocks, setActiveBlocks } = useAppStore()
  const { send } = useEvents()
  const [showAddMenu, setShowAddMenu] = useState(false)
  const [pinnedCollapsed, setPinnedCollapsed] = useState(false)
  const [clipboard, setClipboard] = useState<Record<string, unknown> | null>(null)

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      const oldIdx = activeBlocks.indexOf(active.id as string)
      const newIdx = activeBlocks.indexOf(over.id as string)
      setActiveBlocks(arrayMove(activeBlocks, oldIdx, newIdx))
    }
  }

  const addBlock = (name: string) => {
    if (!activeBlocks.includes(name)) setActiveBlocks([...activeBlocks, name])
    setShowAddMenu(false)
  }

  const removeBlock = (name: string) => {
    setActiveBlocks(activeBlocks.filter(b => b !== name))
  }

  const handleCopy = async () => {
    if (!selectedFaceId) return
    try {
      await api.copyParams(selectedFaceId)
      setClipboard(parameters[selectedFaceId] ?? {})
    } catch { /* ignore */ }
  }

  const handlePaste = async () => {
    if (!selectedFaceId || !clipboard) return
    try { await api.pasteParams(selectedFaceId) } catch { /* ignore */ }
  }

  const handleReset = async () => {
    if (!selectedFaceId) return
    try { await api.resetParams(selectedFaceId) } catch { /* ignore */ }
  }

  // Empty state
  if (!selectedFaceId) {
    return (
      <div className="flex flex-col h-full bg-zinc-900 border-r border-zinc-800 items-center justify-center gap-3">
        <div className="text-4xl">👤</div>
        <p className="text-sm font-medium text-zinc-400">Click on a face to tune</p>
        <p className="text-xs text-zinc-600 text-center px-6">
          Select a face pair in the swap panel to edit its parameters.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-zinc-900 border-r border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Face Options</span>
          <span className="text-xs text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded">
            Face {selectedFaceId.slice(-4)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={handleCopy} title="Copy parameters" className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors">
            <Copy size={12} />
          </button>
          <button onClick={handlePaste} disabled={!clipboard} title="Paste parameters" className="p-1 text-zinc-500 hover:text-zinc-300 disabled:opacity-30 transition-colors">
            <ClipboardPaste size={12} />
          </button>
          <button onClick={handleReset} title="Reset to defaults" className="p-1 text-zinc-500 hover:text-red-400 transition-colors">
            <RotateCcw size={12} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-2">
        {/* Pinned: Face Restorer */}
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg overflow-hidden">
          <div className="flex items-center gap-1.5 px-2 py-1.5 border-b border-zinc-700/50">
            <Pin size={11} className="text-sky-400" />
            <button onClick={() => setPinnedCollapsed(v => !v)} className="flex-1 flex items-center gap-1.5 text-left">
              {pinnedCollapsed ? <ChevronRight size={12} className="text-zinc-500" /> : <ChevronDown size={12} className="text-zinc-500" />}
              <span className="text-xs font-medium text-zinc-300">Face Restorer</span>
            </button>
          </div>
          {!pinnedCollapsed && (
            <div className="p-2">
              <ParameterBlock blockName="Face Restorer" />
            </div>
          )}
        </div>

        {/* Sortable active blocks */}
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
          <SortableContext items={activeBlocks} strategy={verticalListSortingStrategy}>
            {activeBlocks.map(name => (
              <SortableBlock key={name} id={name} onRemove={() => removeBlock(name)} />
            ))}
          </SortableContext>
        </DndContext>

        {/* Add block */}
        <div className="relative">
          <button
            onClick={() => setShowAddMenu(v => !v)}
            className="w-full flex items-center justify-center gap-1.5 py-2 text-xs text-zinc-500 hover:text-zinc-300 border border-dashed border-zinc-700 hover:border-zinc-500 rounded-lg transition-colors"
          >
            <Plus size={12} /> Add Block
          </button>
          {showAddMenu && (
            <div className="absolute bottom-full mb-1 left-0 right-0 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-10 overflow-hidden">
              {ALL_BLOCKS.map(name => {
                const active = activeBlocks.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => !active && addBlock(name)}
                    disabled={active}
                    className={cn(
                      'w-full text-left px-3 py-2 text-xs transition-colors',
                      active ? 'text-zinc-600 cursor-default' : 'text-zinc-300 hover:bg-zinc-700',
                    )}
                  >
                    {active ? '✓ ' : ''}{name}
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
